//! BERT (Bidirectional Encoder Representations from Transformers)
//!
//! Bert is a general large language model that can be used for various language tasks:
//! - Compute sentence embeddings for a prompt.
//! - Compute similarities between a set of sentences.
//! - [Arxiv](https://arxiv.org/abs/1810.04805) "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
//! - Upstream [Github repo](https://github.com/google-research/bert).
//! - See bert in [candle-examples](https://github.com/huggingface/candle/tree/main/candle-examples/) for runnable code
//!
use candle_core::{D, DType, Device, Result, Tensor};
use candle_nn::{Embedding, Module, VarBuilder, embedding};
use candle_transformers::models::with_tracing::{LayerNorm, Linear, layer_norm, linear};
use serde::Deserialize;

pub const DTYPE: DType = DType::F32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HiddenAct {
    Gelu,
    GeluApproximate,
    Relu,
}

#[derive(Clone)]
struct HiddenActLayer {
    act: HiddenAct,
    span: tracing::Span,
}

impl HiddenActLayer {
    fn new(act: HiddenAct) -> Self {
        let span = tracing::span!(tracing::Level::TRACE, "hidden-act");
        Self { act, span }
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        match self.act {
            // https://github.com/huggingface/transformers/blob/cd4584e3c809bb9e1392ccd3fe38b40daba5519a/src/transformers/activations.py#L213
            HiddenAct::Gelu => xs.gelu_erf(),
            HiddenAct::GeluApproximate => xs.gelu(),
            HiddenAct::Relu => xs.relu(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum PositionEmbeddingType {
    #[default]
    Absolute,
}

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/configuration_bert.py#L1
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    pub num_time_series: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub hidden_act: HiddenAct,
    pub hidden_dropout_prob: f64,
    pub max_position_embeddings: usize,
    pub initializer_range: f64,
    pub layer_norm_eps: f64,
    #[serde(default)]
    pub position_embedding_type: PositionEmbeddingType,
    #[serde(default)]
    pub use_cache: bool,
    pub model_type: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            num_time_series: 100,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_act: HiddenAct::Gelu,
            hidden_dropout_prob: 0.1,
            max_position_embeddings: 512,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            position_embedding_type: PositionEmbeddingType::Absolute,
            use_cache: true,
            model_type: Some("bert".to_string()),
        }
    }
}

impl Config {
    fn _all_mini_lm_l6_v2() -> Self {
        // https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/main/config.json
        Self {
            num_time_series: 100,
            hidden_size: 384,
            num_hidden_layers: 6,
            num_attention_heads: 12,
            intermediate_size: 1536,
            hidden_act: HiddenAct::Gelu,
            hidden_dropout_prob: 0.1,
            max_position_embeddings: 512,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            position_embedding_type: PositionEmbeddingType::Absolute,
            use_cache: true,
            model_type: Some("bert".to_string()),
        }
    }
}

#[derive(Clone)]
struct Dropout {
    #[allow(dead_code)]
    pr: f64,
}

impl Dropout {
    fn new(pr: f64) -> Self {
        Self { pr }
    }
}

impl Module for Dropout {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // TODO
        Ok(x.clone())
    }
}

struct FinancialEmbeddings {
    // A Linear layer to project our input vector (e.g., 100 cryptos)
    // into the model's hidden dimension (e.g., 768).
    input_projection: Linear,
    // We keep position_embeddings to give the model a sense of time.
    position_embeddings: Embedding,
    layer_norm: LayerNorm,
    dropout: Dropout,
    span: tracing::Span,
}

impl FinancialEmbeddings {
    // Note the change in the function signature to use our new Config.
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        // The new linear layer.
        let input_projection = linear(
            config.num_time_series,
            config.hidden_size,
            vb.pp("embeddings.input_projection"),
        )?;
        let position_embeddings = embedding(
            config.max_position_embeddings,
            config.hidden_size,
            vb.pp("embeddings.position_embeddings"),
        )?;
        let layer_norm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("embeddings.LayerNorm"),
        )?;
        Ok(Self {
            input_projection,
            position_embeddings,
            layer_norm,
            dropout: Dropout::new(config.hidden_dropout_prob),
            span: tracing::span!(tracing::Level::TRACE, "embeddings"),
        })
    }

    // The forward pass now accepts a float tensor `input_returns`
    // instead of integer `input_ids`.
    fn forward(&self, input_returns: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (_bsize, seq_len, _num_cryptos) = input_returns.dims3()?;

        // Project the input from (batch, seq_len, 100) to (batch, seq_len, 768)
        let input_embeddings = self.input_projection.forward(input_returns)?;

        // The position embedding logic remains the same.
        let position_ids = (0..seq_len as u32).collect::<Vec<_>>();
        let position_ids = Tensor::new(&position_ids[..], input_returns.device())?;
        let position_embeddings = self.position_embeddings.forward(&position_ids)?;

        // Add the projected input and positional info together.
        let embeddings = (&input_embeddings + position_embeddings)?;
        let embeddings = self.layer_norm.forward(&embeddings)?;
        self.dropout.forward(&embeddings)
    }
}

#[derive(Clone)]
struct BertSelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    dropout: Dropout,
    num_attention_heads: usize,
    attention_head_size: usize,
    span: tracing::Span,
    span_softmax: tracing::Span,
}

impl BertSelfAttention {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let attention_head_size = config.hidden_size / config.num_attention_heads;
        let all_head_size = config.num_attention_heads * attention_head_size;
        let dropout = Dropout::new(config.hidden_dropout_prob);
        let hidden_size = config.hidden_size;
        let query = linear(hidden_size, all_head_size, vb.pp("query"))?;
        let value = linear(hidden_size, all_head_size, vb.pp("value"))?;
        let key = linear(hidden_size, all_head_size, vb.pp("key"))?;
        Ok(Self {
            query,
            key,
            value,
            dropout,
            num_attention_heads: config.num_attention_heads,
            attention_head_size,
            span: tracing::span!(tracing::Level::TRACE, "self-attn"),
            span_softmax: tracing::span!(tracing::Level::TRACE, "softmax"),
        })
    }

    fn transpose_for_scores(&self, xs: &Tensor) -> Result<Tensor> {
        let mut new_x_shape = xs.dims().to_vec();
        new_x_shape.pop();
        new_x_shape.push(self.num_attention_heads);
        new_x_shape.push(self.attention_head_size);
        let xs = xs.reshape(new_x_shape.as_slice())?.transpose(1, 2)?;
        xs.contiguous()
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let query_layer = self.query.forward(hidden_states)?;
        let key_layer = self.key.forward(hidden_states)?;
        let value_layer = self.value.forward(hidden_states)?;

        let query_layer = self.transpose_for_scores(&query_layer)?;
        let key_layer = self.transpose_for_scores(&key_layer)?;
        let value_layer = self.transpose_for_scores(&value_layer)?;

        let attention_scores = query_layer.matmul(&key_layer.t()?)?;
        let attention_scores = (attention_scores / (self.attention_head_size as f64).sqrt())?;
        let attention_scores = attention_scores.broadcast_add(attention_mask)?;
        let attention_probs = {
            let _enter_sm = self.span_softmax.enter();
            candle_nn::ops::softmax(&attention_scores, D::Minus1)?
        };
        let attention_probs = self.dropout.forward(&attention_probs)?;

        let context_layer = attention_probs.matmul(&value_layer)?;
        let context_layer = context_layer.transpose(1, 2)?.contiguous()?;
        let context_layer = context_layer.flatten_from(D::Minus2)?;
        Ok(context_layer)
    }
}

#[derive(Clone)]
struct BertSelfOutput {
    dense: Linear,
    layer_norm: LayerNorm,
    dropout: Dropout,
    span: tracing::Span,
}

impl BertSelfOutput {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let dense = linear(config.hidden_size, config.hidden_size, vb.pp("dense"))?;
        let layer_norm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;
        let dropout = Dropout::new(config.hidden_dropout_prob);
        Ok(Self {
            dense,
            layer_norm,
            dropout,
            span: tracing::span!(tracing::Level::TRACE, "self-out"),
        })
    }

    fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let hidden_states = self.dense.forward(hidden_states)?;
        let hidden_states = self.dropout.forward(&hidden_states)?;
        self.layer_norm.forward(&(hidden_states + input_tensor)?)
    }
}

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L392
#[derive(Clone)]
struct BertAttention {
    self_attention: BertSelfAttention,
    self_output: BertSelfOutput,
    span: tracing::Span,
}

impl BertAttention {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let self_attention = BertSelfAttention::load(vb.pp("self"), config)?;
        let self_output = BertSelfOutput::load(vb.pp("output"), config)?;
        Ok(Self {
            self_attention,
            self_output,
            span: tracing::span!(tracing::Level::TRACE, "attn"),
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let self_outputs = self.self_attention.forward(hidden_states, attention_mask)?;
        let attention_output = self.self_output.forward(&self_outputs, hidden_states)?;
        Ok(attention_output)
    }
}

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L441
#[derive(Clone)]
struct BertIntermediate {
    dense: Linear,
    intermediate_act: HiddenActLayer,
    span: tracing::Span,
}

impl BertIntermediate {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let dense = linear(config.hidden_size, config.intermediate_size, vb.pp("dense"))?;
        Ok(Self {
            dense,
            intermediate_act: HiddenActLayer::new(config.hidden_act),
            span: tracing::span!(tracing::Level::TRACE, "inter"),
        })
    }
}

impl Module for BertIntermediate {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let hidden_states = self.dense.forward(hidden_states)?;
        let ys = self.intermediate_act.forward(&hidden_states)?;
        Ok(ys)
    }
}

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L456
#[derive(Clone)]
struct BertOutput {
    dense: Linear,
    layer_norm: LayerNorm,
    dropout: Dropout,
    span: tracing::Span,
}

impl BertOutput {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let dense = linear(config.intermediate_size, config.hidden_size, vb.pp("dense"))?;
        let layer_norm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;
        let dropout = Dropout::new(config.hidden_dropout_prob);
        Ok(Self {
            dense,
            layer_norm,
            dropout,
            span: tracing::span!(tracing::Level::TRACE, "out"),
        })
    }

    fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let hidden_states = self.dense.forward(hidden_states)?;
        let hidden_states = self.dropout.forward(&hidden_states)?;
        self.layer_norm.forward(&(hidden_states + input_tensor)?)
    }
}

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L470
#[derive(Clone)]
pub struct BertLayer {
    attention: BertAttention,
    intermediate: BertIntermediate,
    output: BertOutput,
    span: tracing::Span,
}

impl BertLayer {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let attention = BertAttention::load(vb.pp("attention"), config)?;
        let intermediate = BertIntermediate::load(vb.pp("intermediate"), config)?;
        let output = BertOutput::load(vb.pp("output"), config)?;
        Ok(Self {
            attention,
            intermediate,
            output,
            span: tracing::span!(tracing::Level::TRACE, "layer"),
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let attention_output = self.attention.forward(hidden_states, attention_mask)?;
        // TODO: Support cross-attention?
        // https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L523
        // TODO: Support something similar to `apply_chunking_to_forward`?
        let intermediate_output = self.intermediate.forward(&attention_output)?;
        let layer_output = self
            .output
            .forward(&intermediate_output, &attention_output)?;
        Ok(layer_output)
    }
}

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L556
#[derive(Clone)]
pub struct BertEncoder {
    pub layers: Vec<BertLayer>,
    span: tracing::Span,
}

impl BertEncoder {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let layers = (0..config.num_hidden_layers)
            .map(|index| BertLayer::load(vb.pp(format!("layer.{index}")), config))
            .collect::<Result<Vec<_>>>()?;
        let span = tracing::span!(tracing::Level::TRACE, "encoder");
        Ok(BertEncoder { layers, span })
    }

    pub fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let mut hidden_states = hidden_states.clone();
        // Use a loop rather than a fold as it's easier to modify when adding debug/...
        for layer in self.layers.iter() {
            hidden_states = layer.forward(&hidden_states, attention_mask)?
        }
        Ok(hidden_states)
    }
}

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L874
pub struct FinancialTransformerModel {
    embeddings: FinancialEmbeddings,
    encoder: BertEncoder,
    pub device: Device,
    span: tracing::Span,
}

impl FinancialTransformerModel {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let embeddings = FinancialEmbeddings::load(vb.pp("embeddings"), config)?;
        // The original BertEncoder can be reused without changes.
        let encoder = BertEncoder::load(vb.pp("encoder"), config)?;
        Ok(Self {
            embeddings,
            encoder,
            device: vb.device().clone(),
            span: tracing::span!(tracing::Level::TRACE, "model"),
        })
    }

    pub fn forward(
        &self,
        input_returns: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        println!("input_returns shape: {:?}", input_returns.shape());
        let embedding_output = self.embeddings.forward(input_returns)?;

        let (batch_size, seq_len) = embedding_output.dims2()?;
        let attention_mask = Tensor::ones((batch_size, seq_len), DType::U8, &self.device)?;
        let dtype = embedding_output.dtype();
        
        println!("embedding_output shape: {:?}", embedding_output.shape());
        // https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L995
        let attention_mask = get_extended_attention_mask(&attention_mask, dtype)?;
        println!("attention_mask shape: {:?}", attention_mask.shape());
        let sequence_output = self.encoder.forward(&embedding_output, &attention_mask)?;
        println!("sequence_output shape: {:?}", sequence_output.shape());
        Ok(sequence_output)
    }
}

fn get_extended_attention_mask(attention_mask: &Tensor, dtype: DType) -> Result<Tensor> {
    let attention_mask = match attention_mask.rank() {
        3 => attention_mask.unsqueeze(1)?,
        2 => attention_mask.unsqueeze(1)?.unsqueeze(1)?,
        _ => candle_core::bail!("Wrong shape for input_ids or attention_mask"),
    };
    let attention_mask = attention_mask.to_dtype(dtype)?;
    // torch.finfo(dtype).min
    (attention_mask.ones_like()? - &attention_mask)?.broadcast_mul(
        &Tensor::try_from(f32::MIN)?
            .to_device(attention_mask.device())?
            .to_dtype(dtype)?,
    )
}

//https://github.com/huggingface/transformers/blob/1bd604d11c405dfb8b78bda4062d88fc75c17de0/src/transformers/models/bert/modeling_bert.py#L752-L766
struct BertPredictionHeadTransform {
    dense: Linear,
    activation: HiddenActLayer,
    layer_norm: LayerNorm,
}

impl BertPredictionHeadTransform {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let dense = linear(config.hidden_size, config.hidden_size, vb.pp("dense"))?;
        let activation = HiddenActLayer::new(config.hidden_act);
        let layer_norm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;
        Ok(Self {
            dense,
            activation,
            layer_norm,
        })
    }
}

impl Module for BertPredictionHeadTransform {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let hidden_states = self
            .activation
            .forward(&self.dense.forward(hidden_states)?)?;
        self.layer_norm.forward(&hidden_states)
    }
}

// https://github.com/huggingface/transformers/blob/1bd604d11c405dfb8b78bda4062d88fc75c17de0/src/transformers/models/bert/modeling_bert.py#L769C1-L790C1
pub struct BertLMPredictionHead {
    transform: BertPredictionHeadTransform,
    decoder: Linear,
}

impl BertLMPredictionHead {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let transform = BertPredictionHeadTransform::load(vb.pp("transform"), config)?;
        let decoder = linear(config.hidden_size, config.num_time_series, vb.pp("decoder"))?;
        Ok(Self { transform, decoder })
    }
}

impl Module for BertLMPredictionHead {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        self.decoder
            .forward(&self.transform.forward(hidden_states)?)
    }
}

// https://github.com/huggingface/transformers/blob/1bd604d11c405dfb8b78bda4062d88fc75c17de0/src/transformers/models/bert/modeling_bert.py#L792
pub struct BertOnlyMLMHead {
    predictions: BertLMPredictionHead,
}

impl BertOnlyMLMHead {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let predictions = BertLMPredictionHead::load(vb.pp("predictions"), config)?;
        Ok(Self { predictions })
    }
}

impl Module for BertOnlyMLMHead {
    fn forward(&self, sequence_output: &Tensor) -> Result<Tensor> {
        self.predictions.forward(sequence_output)
    }
}

/* */
pub struct RegressionHead {
    transform: BertPredictionHeadTransform,
    decoder: Linear,
}

impl RegressionHead {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let transform = BertPredictionHeadTransform::load(vb.pp("transform"), config)?;
        // The decoder now outputs `num_cryptos` continuous values
        let decoder = linear(config.hidden_size, config.num_time_series, vb.pp("decoder"))?;
        Ok(Self { transform, decoder })
    }
}

impl Module for RegressionHead {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        self.decoder
            .forward(&self.transform.forward(hidden_states)?)
    }
}
/* */

pub struct FinancialTransformerForMaskedRegression {
    bert: FinancialTransformerModel,
    head: RegressionHead,
}

impl FinancialTransformerForMaskedRegression {
pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let bert = FinancialTransformerModel::load(vb.pp("transformer"), config)?;
        let head = RegressionHead::load(vb.pp("head"), config)?;
        Ok(Self { bert, head })
    }

    pub fn forward(&self, input_returns: &Tensor) -> Result<Tensor> {
        let sequence_output = self.bert.forward(input_returns)?;
        self.head.forward(&sequence_output)
    }
}
