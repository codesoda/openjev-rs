use std::{path::PathBuf, str::FromStr};

use clap::{Args, Parser, Subcommand, ValueEnum};
use openjev_core::PromptProfile;

const ROOT_AFTER_HELP: &str = r#"Examples:
  printf 'ticket body' | openjev --model qwen3-0.6b decide --question 'Which queue?' --option 'Account access' --option Billing
  printf '%s\n' '{"id":"d1","state":"ticket","question":"Which queue?","options":[{"id":"access","description":"Account access"},{"id":"billing","description":"Billing"}]}' | openjev --model qwen3-0.6b ask
  printf '%s\n' '{"id":"d1","state":"ticket","question":"Which queue?","options":[{"id":"access","description":"Account access"},{"id":"billing","description":"Billing"}]}' | openjev --model qwen3-0.6b run
  openjev models list

State text is never trimmed or guessed as JSON. Use --state-json or
--state-json-file for structured state. stdout is JSON/JSONL only.
--compact applies only to decide, noul, score, ask, and run."#;

#[derive(Clone, Debug, Parser)]
#[command(
    name = "openjev",
    version,
    about = "Typed decisions from frozen-model option logits",
    after_help = ROOT_AFTER_HELP
)]
pub struct Cli {
    #[command(flatten)]
    pub global: GlobalArgs,
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Clone, Debug, Default, Args)]
pub struct GlobalArgs {
    #[arg(long, global = true)]
    pub model: Option<String>,
    #[arg(long, global = true)]
    pub model_sha256: Option<String>,
    #[arg(long, global = true, value_parser = parse_profile)]
    pub template_profile: Option<PromptProfile>,
    #[arg(long, global = true)]
    pub cache_dir: Option<PathBuf>,
    #[arg(long, global = true)]
    pub offline: bool,
    #[arg(long, global = true, value_enum)]
    pub device: Option<DeviceArg>,
    #[arg(long, global = true)]
    pub gpu_layers: Option<String>,
    #[arg(long, global = true)]
    pub threads: Option<u32>,
    #[arg(long, global = true)]
    pub n_ctx: Option<u32>,
    #[arg(long, global = true)]
    pub max_tokens: Option<u32>,
    #[arg(long, global = true)]
    pub max_context_tokens: Option<u32>,
    #[arg(long, global = true)]
    pub n_batch: Option<u32>,
    #[arg(long, global = true)]
    pub n_ubatch: Option<u32>,
    #[arg(long, global = true)]
    pub max_sequences: Option<u32>,
    #[arg(long, global = true)]
    pub require_shared: bool,
    #[arg(long, global = true)]
    pub permute: Option<u32>,
    #[arg(long, global = true)]
    pub seed: Option<u64>,
    #[arg(long, global = true, conflicts_with = "calibration")]
    pub temperature: Option<f64>,
    #[arg(long, global = true, conflicts_with = "temperature")]
    pub calibration: Option<PathBuf>,
    #[arg(long, global = true)]
    pub confidence: bool,
    /// Emit the compact decision projection (decide/noul/score/ask/run only).
    #[arg(long, global = true)]
    pub compact: bool,
    #[arg(long, global = true)]
    pub pretty: bool,
    #[arg(long, global = true)]
    pub quiet: bool,
}

#[derive(Clone, Debug, Subcommand)]
pub enum Command {
    /// Score one or more questions against the same state and options.
    Decide(DecideArgs),
    /// Score the fixed yes/no Noul primitive.
    Noul(NoulArgs),
    /// Score ordered finite numeric levels.
    Score(ScoreArgs),
    /// Score one complete Decision JSON object.
    Ask(AskArgs),
    /// Score Decision JSONL in input order.
    Run(RunArgs),
    /// Inspect or populate the verified model cache.
    Models(ModelsArgs),
    /// Evaluate an embedded authored or perturbation fixture.
    Eval(EvalArgs),
    /// Benchmark direct and requested-shared execution.
    Bench(BenchArgs),
    /// Fit temperature calibration (implemented in M7).
    Calibrate(CalibrateArgs),
}

#[derive(Clone, Debug, Args)]
#[command(
    after_help = "Example:\n  printf 'ticket' | openjev decide --question 'Which queue?' --option Access --option Billing"
)]
pub struct DecideArgs {
    #[arg(long, required = true)]
    pub question: Vec<String>,
    #[arg(long, required = true)]
    pub option: Vec<String>,
    #[arg(long)]
    pub option_id: Vec<String>,
    #[command(flatten)]
    pub state: StateArgs,
    #[arg(long)]
    pub id: Option<String>,
}

#[derive(Clone, Debug, Args)]
#[command(
    after_help = "Example:\n  printf 'message' | openjev noul --question 'Is this phishing?'"
)]
pub struct NoulArgs {
    #[arg(long)]
    pub question: String,
    #[command(flatten)]
    pub state: StateArgs,
    #[arg(long)]
    pub id: Option<String>,
}

#[derive(Clone, Debug, Args)]
#[command(
    after_help = "Example:\n  printf 'incident' | openjev score --question 'How urgent?' --level low --level medium --level high"
)]
pub struct ScoreArgs {
    #[arg(long)]
    pub question: String,
    #[arg(long, required = true)]
    pub level: Vec<String>,
    #[arg(long)]
    pub level_id: Vec<String>,
    #[arg(long, allow_hyphen_values = true)]
    pub level_value: Vec<f64>,
    #[command(flatten)]
    pub state: StateArgs,
    #[arg(long)]
    pub id: Option<String>,
}

#[derive(Clone, Debug, Default, Args)]
pub struct StateArgs {
    #[arg(long, conflicts_with_all = ["state_file", "state_json", "state_json_file"])]
    pub state: Option<String>,
    #[arg(long, conflicts_with_all = ["state", "state_json", "state_json_file"])]
    pub state_file: Option<PathBuf>,
    #[arg(long, conflicts_with_all = ["state", "state_file", "state_json_file"])]
    pub state_json: Option<String>,
    #[arg(long, conflicts_with_all = ["state", "state_file", "state_json"])]
    pub state_json_file: Option<PathBuf>,
}

#[derive(Clone, Debug, Args)]
#[command(
    after_help = "Example:\n  printf '%s\\n' '{\"id\":\"d1\",\"state\":\"ticket\",\"question\":\"Queue?\",\"options\":[{\"id\":\"a\",\"description\":\"Access\"},{\"id\":\"b\",\"description\":\"Billing\"}]}' | openjev ask"
)]
pub struct AskArgs {
    #[arg(long, conflicts_with = "input")]
    pub json: Option<String>,
    #[arg(long, conflicts_with = "json")]
    pub input: Option<PathBuf>,
}

#[derive(Clone, Debug, Args)]
#[command(after_help = "Example:\n  openjev run --input decisions.jsonl --output results.jsonl")]
pub struct RunArgs {
    #[arg(long, value_enum, default_value = "direct")]
    pub mode: ModeArg,
    #[arg(long)]
    pub input: Option<PathBuf>,
    #[arg(long)]
    pub output: Option<PathBuf>,
}

#[derive(Clone, Debug, Args)]
#[command(
    after_help = "Examples:\n  openjev models list\n  openjev models pull qwen3-0.6b\n  openjev models path qwen3-0.6b"
)]
pub struct ModelsArgs {
    #[command(subcommand)]
    pub command: Option<ModelsCommand>,
}

#[derive(Clone, Debug, Subcommand)]
pub enum ModelsCommand {
    #[command(after_help = "Example:\n  openjev models list")]
    List,
    #[command(after_help = "Example:\n  openjev models pull qwen3-0.6b --offline")]
    Pull {
        id: String,
        #[arg(long)]
        repair: bool,
    },
    #[command(after_help = "Example:\n  openjev models path qwen3-0.6b")]
    Path { id: String },
    #[command(
        after_help = "Example (M5 surface):\n  openjev models probe qwen3-0.6b --mode shared"
    )]
    Probe {
        id: String,
        #[arg(long, value_enum)]
        mode: ProbeModeArg,
    },
}

#[derive(Clone, Debug, Args)]
#[command(
    after_help = "Examples:\n  openjev eval --fixture authored144 --predictions rows.jsonl\n  openjev --offline eval --fixture authored144 --predictions-output new-rows.jsonl --output new-report.json"
)]
pub struct EvalArgs {
    #[arg(long, value_enum)]
    pub fixture: FixtureArg,
    #[arg(long)]
    pub predictions: Option<PathBuf>,
    #[arg(long, value_enum)]
    pub compare_to: Option<ComparisonArg>,
    /// Create-only raw predictions for an inference-backed evaluation.
    #[arg(long)]
    pub predictions_output: Option<PathBuf>,
    /// Create-only JSON evaluation report. Without it, the report uses stdout.
    #[arg(long)]
    pub output: Option<PathBuf>,
}

#[derive(Clone, Debug, Args)]
#[command(
    after_help = "Example:\n  openjev bench --state-file fixtures/bench/long-state.txt --questions fixtures/bench/questions21.jsonl --samples-output new-samples.jsonl --output new-report.json"
)]
pub struct BenchArgs {
    #[arg(long)]
    pub state_file: PathBuf,
    #[arg(long)]
    pub questions: PathBuf,
    #[arg(long, default_value_t = 5)]
    pub repeats: u32,
    #[arg(long)]
    pub output: Option<PathBuf>,
    /// Optional create-only JSONL file containing one row per timed sample.
    #[arg(long)]
    pub samples_output: Option<PathBuf>,
}

#[derive(Clone, Debug, Args)]
#[command(
    after_help = "Example (M7 surface):\n  openjev calibrate --input labelled-logits.jsonl --output calibration.json"
)]
pub struct CalibrateArgs {
    #[arg(long)]
    pub input: PathBuf,
    #[arg(long)]
    pub output: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum DeviceArg {
    Cpu,
    Metal,
    Cuda,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum ModeArg {
    Direct,
    Serial,
    Shared,
    Batch,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum ProbeModeArg {
    Shared,
    Batch,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum FixtureArg {
    Authored144,
    Perturbations108,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum ComparisonArg {
    BrowserLadder,
}

fn parse_profile(value: &str) -> Result<PromptProfile, String> {
    PromptProfile::from_str(value).map_err(|error| error.to_string())
}
