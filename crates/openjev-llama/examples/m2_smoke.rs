#[cfg(feature = "native")]
mod native {
    use std::{io::Write as _, path::PathBuf, str::FromStr as _};

    use openjev_core::{Decision, DecisionOption, Device, GpuLayersRequested, StateValue};
    use openjev_llama::{
        CacheOptions, EngineHandle, EngineOptions, ModelCache, ModelRegistry, TemplateStatus,
    };
    use serde::Serialize;
    use tracing_subscriber::EnvFilter;

    #[derive(Debug)]
    struct Args {
        model: Option<String>,
        all: bool,
        cache_dir: Option<PathBuf>,
        offline: bool,
        repair: bool,
        device: Device,
        gpu_layers: GpuLayersRequested,
        threads: Option<u32>,
        n_ctx: Option<u32>,
    }

    #[derive(Serialize)]
    struct SuccessRow<T: Serialize> {
        schema: &'static str,
        model_id: String,
        outcome: &'static str,
        artifact: openjev_llama::VerifiedArtifact,
        smoke: T,
    }

    #[derive(Serialize)]
    struct ErrorRow {
        schema: &'static str,
        model_id: String,
        outcome: &'static str,
        error: String,
    }

    pub fn run() -> i32 {
        if std::env::var("OPENJEV_INTEGRATION").as_deref() != Ok("1") {
            let row = ErrorRow {
                schema: "openjev-m2-smoke-error-v1",
                model_id: String::new(),
                outcome: "failed",
                error: "m2_smoke requires OPENJEV_INTEGRATION=1".to_owned(),
            };
            let _ = write_json(&row);
            return 2;
        }
        tracing_subscriber::fmt()
            .with_env_filter(
                EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
            )
            .with_ansi(false)
            .with_writer(std::io::stderr)
            .init();

        let args = match parse_args() {
            Ok(args) => args,
            Err(error) => {
                let row = ErrorRow {
                    schema: "openjev-m2-smoke-error-v1",
                    model_id: String::new(),
                    outcome: "failed",
                    error,
                };
                let _ = write_json(&row);
                return 2;
            }
        };
        let registry = match ModelRegistry::bundled() {
            Ok(registry) => registry,
            Err(error) => {
                let row = ErrorRow {
                    schema: "openjev-m2-smoke-error-v1",
                    model_id: String::new(),
                    outcome: "failed",
                    error: error.to_string(),
                };
                let _ = write_json(&row);
                return 1;
            }
        };
        let cache = match ModelCache::from_precedence(args.cache_dir.as_deref()) {
            Ok(cache) => cache,
            Err(error) => {
                let row = ErrorRow {
                    schema: "openjev-m2-smoke-error-v1",
                    model_id: String::new(),
                    outcome: "failed",
                    error: error.to_string(),
                };
                let _ = write_json(&row);
                return 1;
            }
        };
        tracing::info!(cache_root = %cache.root().display(), "using canonical openjev cache");

        let ids: Vec<String> = if args.all {
            registry
                .list()
                .iter()
                .map(|model| model.id.clone())
                .collect()
        } else {
            vec![
                args.model
                    .clone()
                    .unwrap_or_else(|| registry.default_model().to_owned()),
            ]
        };
        let mut failed = false;
        for id in ids {
            let result = run_one(&registry, &cache, &args, &id);
            match result {
                Ok((artifact, smoke)) => {
                    let outcome = match smoke.model.template_status {
                        TemplateStatus::Exact
                        | TemplateStatus::ReviewedEquivalent
                        | TemplateStatus::OverrideUnverified => "passed",
                        TemplateStatus::Mismatch | TemplateStatus::Missing => {
                            failed = true;
                            "needs-template-adjudication"
                        }
                    };
                    let row = SuccessRow {
                        schema: "openjev-m2-smoke-row-v1",
                        model_id: id,
                        outcome,
                        artifact,
                        smoke,
                    };
                    if write_json(&row).is_err() {
                        return 1;
                    }
                }
                Err(error) => {
                    failed = true;
                    tracing::error!(model = %id, error = %error, "M2 model smoke failed; continuing");
                    let row = ErrorRow {
                        schema: "openjev-m2-smoke-error-v1",
                        model_id: id,
                        outcome: "failed",
                        error,
                    };
                    if write_json(&row).is_err() {
                        return 1;
                    }
                }
            }
        }
        i32::from(failed)
    }

    fn run_one(
        registry: &ModelRegistry,
        cache: &ModelCache,
        args: &Args,
        id: &str,
    ) -> Result<
        (
            openjev_llama::VerifiedArtifact,
            openjev_llama::DirectSmokeReport,
        ),
        String,
    > {
        let spec = registry
            .resolve(id)
            .map_err(|error| error.to_string())?
            .clone();
        tracing::info!(model = %id, repo = %spec.repo, revision = %spec.revision, file = %spec.file, "resolving pinned artifact");
        let artifact = cache
            .ensure(
                &spec,
                CacheOptions {
                    offline: args.offline,
                    repair: args.repair,
                },
            )
            .map_err(|error| error.to_string())?;
        tracing::info!(model = %id, path = %artifact.path.display(), cache_hit = artifact.cache_hit, "verified artifact SHA-256 and size");
        let mut options = EngineOptions {
            device: args.device,
            gpu_layers: args.gpu_layers,
            n_ctx: args.n_ctx,
            ..EngineOptions::default()
        };
        if let Some(threads) = args.threads {
            options.threads = threads;
        }
        let engine = EngineHandle::spawn(spec, artifact.clone(), options)
            .map_err(|error| error.to_string())?;
        let decision = Decision::new(
            "m2-smoke",
            StateValue::string("A local smoke row for exact slot and boundary validation.")
                .map_err(|error| error.to_string())?,
            "Which option best identifies this test?",
            vec![
                DecisionOption {
                    id: "load".to_owned(),
                    description: "Model load smoke".to_owned(),
                },
                DecisionOption {
                    id: "generate".to_owned(),
                    description: "Text generation".to_owned(),
                },
                DecisionOption {
                    id: "train".to_owned(),
                    description: "Model training".to_owned(),
                },
            ],
        )
        .map_err(|error| error.to_string())?;
        let report = engine
            .smoke_direct(decision)
            .map_err(|error| error.to_string())?;
        engine.shutdown().map_err(|error| error.to_string())?;
        Ok((artifact, report))
    }

    fn write_json<T: Serialize>(value: &T) -> Result<(), String> {
        let stdout = std::io::stdout();
        let mut lock = stdout.lock();
        serde_json::to_writer(&mut lock, value).map_err(|error| error.to_string())?;
        lock.write_all(b"\n").map_err(|error| error.to_string())
    }

    fn parse_args() -> Result<Args, String> {
        let mut model = None;
        let mut all = false;
        let mut cache_dir = None;
        let mut offline = false;
        let mut repair = false;
        let mut device = Device::Cpu;
        let mut gpu_layers = GpuLayersRequested::Count(0);
        let mut threads = None;
        let mut n_ctx = None;
        let mut arguments = std::env::args().skip(1);
        while let Some(argument) = arguments.next() {
            match argument.as_str() {
                "--model" => model = Some(next_value(&mut arguments, "--model")?),
                "--all" => all = true,
                "--cache-dir" => {
                    cache_dir = Some(PathBuf::from(next_value(&mut arguments, "--cache-dir")?));
                }
                "--offline" => offline = true,
                "--repair" => repair = true,
                "--device" => {
                    let value = next_value(&mut arguments, "--device")?;
                    device = match value.as_str() {
                        "cpu" => Device::Cpu,
                        "metal" => Device::Metal,
                        "cuda" => Device::Cuda,
                        _ => return Err(format!("invalid --device {value:?}")),
                    };
                }
                "--gpu-layers" => {
                    let value = next_value(&mut arguments, "--gpu-layers")?;
                    gpu_layers = if value == "all" {
                        GpuLayersRequested::All
                    } else {
                        GpuLayersRequested::Count(
                            u32::from_str(&value)
                                .map_err(|error| format!("invalid --gpu-layers: {error}"))?,
                        )
                    };
                }
                "--threads" => {
                    threads = Some(parse_u32(
                        next_value(&mut arguments, "--threads")?,
                        "threads",
                    )?);
                }
                "--n-ctx" => {
                    n_ctx = Some(parse_u32(next_value(&mut arguments, "--n-ctx")?, "n-ctx")?);
                }
                "--help" => {
                    return Err("usage: m2_smoke [--all | --model ID] [--cache-dir PATH] [--offline] [--repair] --device cpu|metal|cuda --gpu-layers all|N [--threads N] [--n-ctx N]".to_owned());
                }
                _ => return Err(format!("unknown argument {argument:?}")),
            }
        }
        if all && model.is_some() {
            return Err("--all and --model are mutually exclusive".to_owned());
        }
        match device {
            Device::Cpu => gpu_layers = GpuLayersRequested::Count(0),
            Device::Metal | Device::Cuda if gpu_layers == GpuLayersRequested::Count(0) => {
                gpu_layers = GpuLayersRequested::All;
            }
            Device::Metal | Device::Cuda => {}
        }
        Ok(Args {
            model,
            all,
            cache_dir,
            offline,
            repair,
            device,
            gpu_layers,
            threads,
            n_ctx,
        })
    }

    fn next_value(
        arguments: &mut impl Iterator<Item = String>,
        flag: &str,
    ) -> Result<String, String> {
        arguments
            .next()
            .ok_or_else(|| format!("{flag} requires a value"))
    }

    fn parse_u32(value: String, name: &str) -> Result<u32, String> {
        u32::from_str(&value).map_err(|error| format!("invalid --{name}: {error}"))
    }
}

#[cfg(feature = "native")]
fn main() {
    std::process::exit(native::run());
}

#[cfg(not(feature = "native"))]
fn main() {
    eprintln!(
        "{{\"schema\":\"openjev-error-v1\",\"error\":{{\"code\":\"backend_unavailable\",\"message\":\"m2_smoke requires --features native, metal, or cuda\",\"details\":{{}}}}}}"
    );
    std::process::exit(1);
}
