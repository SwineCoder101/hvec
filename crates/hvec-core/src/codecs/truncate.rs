//! Matryoshka-style truncation, optionally followed by another codec.

use crate::{Codec, CodecError, Encoded, Metric, Vector};

use super::check_dim;

/// Keep the first `keep` dimensions and hand them to an inner codec.
///
/// Embedding models trained with Matryoshka representation learning put the
/// most information in the leading dimensions, so a prefix of the vector is a
/// usable embedding on its own. On a model not trained that way, the same
/// truncation is a plain lossy projection. Composing with an inner codec
/// (`mrl256+int8`) gives two independent levers over the byte budget.
///
/// Layout and parameters: the inner codec's. Names: `mrl<k>` when the inner
/// codec is `f32`, otherwise `mrl<k>+<inner>`.
///
/// Error model: the inner codec's, on the truncated space. `score` truncates
/// the query too, so `cosine` is the cosine of the two prefixes, which is the
/// Matryoshka convention, and `L2` omits the query's dropped tail. Both
/// differ from the reference distance on the zero-padded decoded vector by a
/// per-query constant (a factor for cosine, an offset for L2), so rankings
/// are the same.
#[derive(Debug)]
pub struct TruncateCodec {
    dimension: usize,
    keep: usize,
    name: String,
    inner: Box<dyn Codec>,
}

impl TruncateCodec {
    /// `inner` must have dimension `keep`, with `1 <= keep <= dimension`.
    pub fn new(dimension: usize, keep: usize, inner: Box<dyn Codec>) -> Result<Self, CodecError> {
        let name = if inner.name() == "f32" {
            format!("mrl{keep}")
        } else {
            format!("mrl{keep}+{}", inner.name())
        };
        if keep == 0 || keep > dimension {
            return Err(CodecError::BadParams {
                codec: name,
                reason: format!("cannot keep {keep} of {dimension} dimensions"),
            });
        }
        if inner.dimension() != keep {
            return Err(CodecError::BadParams {
                codec: name,
                reason: format!("inner codec has dimension {}, expected {keep}", inner.dimension()),
            });
        }
        Ok(Self {
            dimension,
            keep,
            name,
            inner,
        })
    }

    /// Dimensions kept.
    #[must_use]
    pub fn keep(&self) -> usize {
        self.keep
    }

    fn prefix<'a>(&self, vector: &'a [f32]) -> Result<&'a [f32], CodecError> {
        check_dim(self.dimension, vector.len())?;
        Ok(&vector[..self.keep])
    }
}

impl Codec for TruncateCodec {
    fn name(&self) -> &str {
        &self.name
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn encoded_size(&self) -> usize {
        self.inner.encoded_size()
    }

    fn supported_metrics(&self) -> &'static [Metric] {
        self.inner.supported_metrics()
    }

    fn needs_fit(&self) -> bool {
        self.inner.needs_fit()
    }

    fn fit(&mut self, sample: &[Vector]) -> Result<(), CodecError> {
        let truncated: Vec<Vector> = sample
            .iter()
            .map(|v| self.prefix(v).map(<[f32]>::to_vec))
            .collect::<Result<_, _>>()?;
        self.inner.fit(&truncated)
    }

    fn params(&self) -> Vec<u8> {
        self.inner.params()
    }

    fn restore(&mut self, params: &[u8]) -> Result<(), CodecError> {
        self.inner.restore(params)
    }

    fn encode(&self, vector: &[f32]) -> Result<Encoded, CodecError> {
        self.inner.encode(self.prefix(vector)?)
    }

    fn decode(&self, encoded: &Encoded) -> Result<Vector, CodecError> {
        let mut v = self.inner.decode(encoded)?;
        v.resize(self.dimension, 0.0);
        Ok(v)
    }

    fn score(&self, query: &[f32], encoded: &Encoded, metric: Metric) -> Result<f32, CodecError> {
        self.inner.score(self.prefix(query)?, encoded, metric)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codecs::{BinaryCentredCodec, F32Codec, Int8Codec};
    use crate::distance;

    #[test]
    fn validates_shape_and_names_itself() {
        assert!(matches!(
            TruncateCodec::new(8, 0, Box::new(F32Codec::new(0))),
            Err(CodecError::BadParams { .. })
        ));
        assert!(matches!(
            TruncateCodec::new(8, 9, Box::new(F32Codec::new(9))),
            Err(CodecError::BadParams { .. })
        ));
        assert!(matches!(
            TruncateCodec::new(8, 4, Box::new(F32Codec::new(8))),
            Err(CodecError::BadParams { .. })
        ));
        let plain = TruncateCodec::new(8, 4, Box::new(F32Codec::new(4))).unwrap();
        assert_eq!(plain.name(), "mrl4");
        assert_eq!(plain.encoded_size(), 16);
        assert_eq!(plain.compression_ratio(), 2.0);
        let i8 = TruncateCodec::new(8, 4, Box::new(Int8Codec::new(4))).unwrap();
        assert_eq!(i8.name(), "mrl4+int8");
        assert_eq!(i8.encoded_size(), 12);
        assert_eq!(i8.keep(), 4);
    }

    #[test]
    fn scores_on_the_prefix_and_pads_on_decode() {
        let c = TruncateCodec::new(6, 3, Box::new(F32Codec::new(3))).unwrap();
        let x = [1.0, 2.0, 3.0, 100.0, 100.0, 100.0];
        let q = [1.0, 0.0, 0.0, -50.0, 0.0, 0.0];
        let enc = c.encode(&x).unwrap();
        assert_eq!(enc.0.len(), 12);
        assert_eq!(c.decode(&enc).unwrap(), vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0]);
        assert_eq!(c.score(&q, &enc, Metric::Dot).unwrap(), 1.0);
        assert!((c.score(&q, &enc, Metric::Cosine).unwrap() - distance::cosine(&q[..3], &x[..3])).abs() < 1e-6);
        assert_eq!(
            c.score(&q, &enc, Metric::L2).unwrap(),
            distance::l2_squared(&q[..3], &x[..3])
        );
        assert!(matches!(c.encode(&[0.0; 5]), Err(CodecError::DimensionMismatch { .. })));
        assert!(matches!(
            c.score(&[0.0; 7], &enc, Metric::Dot),
            Err(CodecError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn delegates_fitting_and_parameters_to_the_inner_codec() {
        let mut c = TruncateCodec::new(4, 2, Box::new(BinaryCentredCodec::new(2))).unwrap();
        assert!(c.needs_fit());
        assert_eq!(c.name(), "mrl2+binary-centred");
        c.fit(&[vec![1.0, 0.0, 9.0, 9.0], vec![3.0, 2.0, 9.0, 9.0]]).unwrap();
        assert!(!c.needs_fit());
        let params = c.params();
        assert_eq!(params.len(), 12, "mean of 2 dims plus a scale");
        let restored = crate::codec_with_params("mrl2+binary-centred", 4, &params).unwrap();
        assert_eq!(restored.params(), params);
        let v = [2.5, 0.5, -1.0, -1.0];
        assert_eq!(restored.encode(&v).unwrap(), c.encode(&v).unwrap());
        assert!(matches!(
            crate::codec_with_params("mrl2+binary-centred", 4, &[]),
            Err(CodecError::BadParams { .. })
        ));
        assert!(matches!(
            crate::codec_with_params("mrl2", 4, &params),
            Err(CodecError::BadParams { .. })
        ));
    }
}
