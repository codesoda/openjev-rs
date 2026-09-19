//! Backend-neutral types and algorithms for openjev.

pub mod eval;
pub mod numerics;
pub mod primitives;
pub mod prompt;
pub mod slots;
pub mod types;
pub mod validate;

pub use eval::{EvalReport, EvalSummary, GoldRow, Prediction, ProbabilityInput, evaluate};
pub use numerics::{NumericReadout, first_argmax, normalized_margin, read_logits, softmax};
pub use primitives::{Noul, Score, ScoreLevel};
pub use prompt::{
    DIRECT_SYSTEM, PROMPT_VERSION, PreparedPrompt, PromptProfile, prepare_prompt,
    python_json_dumps, state_prefix_text,
};
pub use slots::{LETTERS, SlotTokenizer, VerifiedSlots, verify_slots};
pub use types::{
    CONDITIONAL_PROBABILITY_LIMITATION, CONFIDENCE_STATUS, DIRECT_READOUT, Decision,
    DecisionOption, Device, ErrorDetail, ErrorRecord, ExecutionMetadata, ExecutionMode,
    FORCED_TYPED_LIMITATION, GpuLayersRequested, GpuLayersStatus, Integrity, ModelMetadata,
    NativeReference, OpenJevError, PROBABILITY_STATUS, Postprocess, Primitive, Question, RawSample,
    Readout, SharedTiming, StateValue, TemplateMetadataStatus, standard_limitations,
};
pub use validate::{MAX_JSON_DEPTH, parse_json_strict};
