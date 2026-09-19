use std::{path::PathBuf, str::FromStr};

use clap::{Args, Parser, Subcommand, ValueEnum};
use openjev_core::PromptProfile;

#[derive(Clone, Debug, Parser)]
#[command(
    name = "openjev",
    version,
    about = "Typed decisions from frozen-model option logits"
)]
pub struct Cli {
    #[command(flatten)]
    pub global: GlobalArgs,
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Clone, Debug, Args)]
pub struct GlobalArgs {
    #[arg(long, global = true, default_value = "minicpm5-2b")]
    pub model: String,
    #[arg(long, global = true)]
    pub model_sha256: Option<String>,
    #[arg(long, global = true, value_parser = parse_profile)]
    pub template_profile: Option<PromptProfile>,
    #[arg(long, global = true)]
    pub cache_dir: Option<PathBuf>,
    #[arg(long, global = true)]
    pub offline: bool,
    #[arg(long, global = true, value_enum, default_value = "cpu")]
    pub device: DeviceArg,
    #[arg(long, global = true, default_value = "all")]
    pub gpu_layers: String,
    #[arg(long, global = true)]
    pub threads: Option<u32>,
    #[arg(long, global = true)]
    pub n_ctx: Option<u32>,
    #[arg(long, global = true, default_value_t = 4096)]
    pub max_tokens: u32,
    #[arg(long, global = true, default_value_t = 32768)]
    pub max_context_tokens: u32,
    #[arg(long, global = true, default_value_t = 512)]
    pub n_batch: u32,
    #[arg(long, global = true, default_value_t = 512)]
    pub n_ubatch: u32,
    #[arg(long, global = true, default_value_t = 32)]
    pub max_sequences: u32,
    #[arg(long, global = true)]
    pub require_shared: bool,
    #[arg(long, global = true, default_value_t = 1)]
    pub permute: u32,
    #[arg(long, global = true, default_value_t = 0)]
    pub seed: u64,
    #[arg(long, global = true, conflicts_with = "calibration")]
    pub temperature: Option<f64>,
    #[arg(long, global = true, conflicts_with = "temperature")]
    pub calibration: Option<PathBuf>,
    #[arg(long, global = true)]
    pub confidence: bool,
    #[arg(long, global = true)]
    pub pretty: bool,
    #[arg(long, global = true)]
    pub quiet: bool,
}

#[derive(Clone, Debug, Subcommand)]
pub enum Command {
    Decide(DecideArgs),
    Noul(NoulArgs),
    Score(ScoreArgs),
    Ask(AskArgs),
    Run(RunArgs),
    Models(ModelsArgs),
    Eval(EvalArgs),
    Bench(BenchArgs),
    Calibrate(CalibrateArgs),
}

#[derive(Clone, Debug, Args)]
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
pub struct NoulArgs {
    #[arg(long)]
    pub question: String,
    #[command(flatten)]
    pub state: StateArgs,
    #[arg(long)]
    pub id: Option<String>,
}

#[derive(Clone, Debug, Args)]
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

#[derive(Clone, Debug, Args)]
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
pub struct AskArgs {
    #[arg(long, conflicts_with = "input")]
    pub json: Option<String>,
    #[arg(long, conflicts_with = "json")]
    pub input: Option<PathBuf>,
}

#[derive(Clone, Debug, Args)]
pub struct RunArgs {
    #[arg(long, value_enum, default_value = "direct")]
    pub mode: ModeArg,
    #[arg(long)]
    pub input: Option<PathBuf>,
    #[arg(long)]
    pub output: Option<PathBuf>,
}

#[derive(Clone, Debug, Args)]
pub struct ModelsArgs {
    #[command(subcommand)]
    pub command: Option<ModelsCommand>,
}

#[derive(Clone, Debug, Subcommand)]
pub enum ModelsCommand {
    List,
    Pull {
        id: String,
        #[arg(long)]
        repair: bool,
    },
    Path {
        id: String,
    },
    Probe {
        id: String,
        #[arg(long, value_enum)]
        mode: ProbeModeArg,
    },
}

#[derive(Clone, Debug, Args)]
pub struct EvalArgs {
    #[arg(long, value_enum)]
    pub fixture: FixtureArg,
    #[arg(long)]
    pub predictions: Option<PathBuf>,
    #[arg(long, value_enum)]
    pub compare_to: Option<ComparisonArg>,
}

#[derive(Clone, Debug, Args)]
pub struct BenchArgs {
    #[arg(long)]
    pub state_file: PathBuf,
    #[arg(long)]
    pub questions: PathBuf,
    #[arg(long, default_value_t = 5)]
    pub repeats: u32,
    #[arg(long)]
    pub output: Option<PathBuf>,
}

#[derive(Clone, Debug, Args)]
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
