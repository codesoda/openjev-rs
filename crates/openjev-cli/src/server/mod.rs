pub mod jev;

use std::{
    net::{IpAddr, Ipv4Addr, SocketAddr},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc::{self, Receiver, SyncSender, TrySendError},
    },
    thread::JoinHandle,
    time::{Duration, Instant},
};

use axum::{
    Json, Router,
    body::to_bytes,
    extract::{Request, State},
    http::{HeaderMap, HeaderValue, Method, StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use openjev_core::{Decision, DecisionOption, ExecutionMode, ModelMetadata, StateValue};
use serde::Serialize;
use serde_json::Value;
use tokio::sync::{Semaphore, oneshot};

use crate::{
    CliError,
    args::GlobalArgs,
    attempt_native_group,
    commands::{self, DecisionScorer, ScoringConfig},
};

const DEFAULT_HOST: IpAddr = IpAddr::V4(Ipv4Addr::LOCALHOST);
const DEFAULT_PORT: u16 = 8080;
const DEFAULT_REQUEST_TIMEOUT_SECS: u64 = 120;
const MAX_BODY_BYTES: usize = 1024 * 1024;
const MAX_ADMITTED_JOBS: usize = 16;
const RETRY_AFTER_SECS: &str = "1";
const PROBABILITY_HEADER: &str = "conditional option score; uncalibrated as decision confidence";

#[derive(Clone, Debug)]
pub struct ServeOptions {
    pub host: IpAddr,
    pub port: u16,
    pub request_timeout: Duration,
    api_key: Option<Arc<str>>,
}

impl ServeOptions {
    pub fn from_cli(
        host: Option<IpAddr>,
        port: Option<u16>,
        timeout_secs: Option<u64>,
        api_key_env: Option<&str>,
    ) -> Result<Self, CliError> {
        let host = host.unwrap_or(DEFAULT_HOST);
        let port = port.unwrap_or(DEFAULT_PORT);
        if port == 0 {
            return Err(CliError::validation("--port must be positive"));
        }
        let timeout_secs = timeout_secs.unwrap_or(DEFAULT_REQUEST_TIMEOUT_SECS);
        if timeout_secs == 0 {
            return Err(CliError::validation(
                "--request-timeout-secs must be positive",
            ));
        }
        let request_timeout = Duration::from_secs(timeout_secs);
        if Instant::now().checked_add(request_timeout).is_none() {
            return Err(CliError::validation(
                "--request-timeout-secs is too large for this platform",
            ));
        }
        let api_key = api_key_env
            .map(|name| {
                if name.is_empty() || name.contains('=') {
                    return Err(CliError::validation(
                        "--api-key-env must name a nonempty environment variable",
                    ));
                }
                let value = std::env::var(name).map_err(|_| {
                    CliError::validation(format!(
                        "--api-key-env variable {name:?} is missing or not valid UTF-8"
                    ))
                })?;
                if value.trim().is_empty() {
                    return Err(CliError::validation(format!(
                        "--api-key-env variable {name:?} must contain a nonempty bearer secret"
                    )));
                }
                HeaderValue::from_str(&format!("Bearer {value}")).map_err(|_| {
                    CliError::validation("bearer secret is not valid in an HTTP header")
                })?;
                Ok(Arc::<str>::from(value))
            })
            .transpose()?;
        if !host.is_loopback() && api_key.is_none() {
            return Err(CliError::validation(
                "non-loopback --host requires --api-key-env with a nonempty bearer secret",
            ));
        }
        Ok(Self {
            host,
            port,
            request_timeout,
            api_key,
        })
    }
}

pub fn validate_server_global_args(global: &GlobalArgs) -> Result<(), CliError> {
    if global.compact
        || global.pretty
        || global.confidence
        || global.permute.is_some()
        || global.seed.is_some()
        || global.temperature.is_some()
        || global.calibration.is_some()
    {
        return Err(CliError::validation(
            "--compact, --pretty, --confidence, --permute, --seed, --temperature, and --calibration are not valid with --serve",
        ));
    }
    Ok(())
}

pub fn run(
    config: ScoringConfig,
    options: ServeOptions,
    require_shared: bool,
) -> Result<(), CliError> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|error| CliError::runtime("server_runtime", error.to_string()))?;
    runtime.block_on(run_async(config, options, require_shared))
}

async fn run_async(
    config: ScoringConfig,
    options: ServeOptions,
    require_shared: bool,
) -> Result<(), CliError> {
    let address = SocketAddr::new(options.host, options.port);
    let listener = tokio::net::TcpListener::bind(address)
        .await
        .map_err(|error| {
            CliError::runtime("server_bind", format!("cannot bind {address}: {error}"))
        })?;
    let shutdown = Arc::new(AtomicBool::new(false));
    let factory_config = config.clone();
    let (worker, startup) = WorkerHandle::spawn(
        move || crate::load_scorer(&factory_config),
        config.max_sequences,
        require_shared,
        Arc::clone(&shutdown),
    )?;
    let ready = worker.ready();
    let state = AppState {
        sender: worker.sender(),
        admission: Arc::new(Semaphore::new(MAX_ADMITTED_JOBS)),
        ready: Arc::clone(&ready),
        shutdown: Arc::clone(&shutdown),
        request_timeout: options.request_timeout,
        api_key: options.api_key,
        model: startup,
        request_sequence: Arc::new(AtomicU64::new(1)),
    };
    let app = router(state.clone());
    tracing::info!(
        address = %address,
        model = %state.model.public_id,
        "openjev server ready after one disclosed warmup decision"
    );
    let ready_for_signal = Arc::clone(&ready);
    let shutdown_for_signal = Arc::clone(&shutdown);
    let serve_result = axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            shutdown_signal().await;
            shutdown_for_signal.store(true, Ordering::Release);
            ready_for_signal.store(false, Ordering::Release);
            tracing::info!("openjev server stopping admission; waiting for in-flight native work");
        })
        .await;
    shutdown.store(true, Ordering::Release);
    ready.store(false, Ordering::Release);
    let worker_result = worker.shutdown();
    serve_result.map_err(|error| CliError::runtime("server_io", error.to_string()))?;
    worker_result
}

async fn shutdown_signal() {
    #[cfg(unix)]
    {
        if let Ok(mut terminate) =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        {
            tokio::select! {
                _ = tokio::signal::ctrl_c() => {},
                _ = terminate.recv() => {},
            }
        } else {
            let _ = tokio::signal::ctrl_c().await;
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}

#[derive(Clone)]
struct AppState {
    sender: SyncSender<WorkerCommand>,
    admission: Arc<Semaphore>,
    ready: Arc<AtomicBool>,
    shutdown: Arc<AtomicBool>,
    request_timeout: Duration,
    api_key: Option<Arc<str>>,
    model: StartupInfo,
    request_sequence: Arc<AtomicU64>,
}

#[derive(Clone, Debug)]
struct StartupInfo {
    public_id: String,
    description: String,
    release_date: &'static str,
}

fn router(state: AppState) -> Router {
    Router::new()
        .route("/v1/systemone", post(system_one))
        .route("/v1/models", get(models))
        .route("/healthz", get(health))
        .route("/readyz", get(ready))
        .method_not_allowed_fallback(method_not_allowed)
        .fallback(not_found)
        .with_state(state)
}

async fn health() -> Response {
    json_value(StatusCode::OK, serde_json::json!({"status":"ok"}))
}

async fn ready(State(state): State<AppState>) -> Response {
    if state.ready.load(Ordering::Acquire) && !state.shutdown.load(Ordering::Acquire) {
        json_value(StatusCode::OK, serde_json::json!({"status":"ready"}))
    } else {
        api_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "not_ready",
            "the resident inference worker is not ready",
        )
    }
}

#[derive(Serialize)]
struct ModelsEnvelope<'a> {
    models: [ModelCard<'a>; 1],
}

#[derive(Serialize)]
struct ModelCard<'a> {
    name: &'a str,
    description: &'a str,
    release_date: &'a str,
}

async fn models(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Some(response) = authorization_error(&state, &headers) {
        return response;
    }
    json_response(
        StatusCode::OK,
        &ModelsEnvelope {
            models: [ModelCard {
                name: &state.model.public_id,
                description: &state.model.description,
                release_date: state.model.release_date,
            }],
        },
    )
}

async fn system_one(State(state): State<AppState>, request: Request) -> Response {
    let started = Instant::now();
    if state.shutdown.load(Ordering::Acquire) || !state.ready.load(Ordering::Acquire) {
        return api_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "not_ready",
            "the resident inference worker is not ready",
        );
    }
    if let Some(response) = authorization_error(&state, request.headers()) {
        return response;
    }
    if !has_json_content_type(request.headers()) {
        return api_error(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "unsupported_media_type",
            "Content-Type must be application/json",
        );
    }
    let permit = match Arc::clone(&state.admission).try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => return overload(),
    };
    let deadline = started + state.request_timeout;
    let remaining = match deadline.checked_duration_since(Instant::now()) {
        Some(remaining) if !remaining.is_zero() => remaining,
        _ => return timeout_response(),
    };
    let bytes = match tokio::time::timeout(remaining, to_bytes(request.into_body(), MAX_BODY_BYTES))
        .await
    {
        Err(_) => return timeout_response(),
        Ok(Err(error)) => {
            tracing::debug!(error = %error, "HTTP request body rejected");
            if std::error::Error::source(&error)
                .is_some_and(|source| source.is::<http_body_util::LengthLimitError>())
            {
                return api_error(
                    StatusCode::PAYLOAD_TOO_LARGE,
                    "body_too_large",
                    "request body exceeds the 1 MiB limit",
                );
            }
            return api_error(
                StatusCode::BAD_REQUEST,
                "body_read_error",
                "request body could not be read",
            );
        }
        Ok(Ok(bytes)) => bytes,
    };
    let prepared = match jev::parse_request(&bytes) {
        Ok(prepared) => prepared,
        Err(error) => {
            let status = if error.error_type == "invalid_json" {
                StatusCode::BAD_REQUEST
            } else {
                StatusCode::UNPROCESSABLE_ENTITY
            };
            return api_error(status, error.error_type, &error.message);
        }
    };
    if let Some(requested) = &prepared.requested_model
        && requested != "jev-latest"
        && requested != &state.model.public_id
    {
        return api_error(
            StatusCode::NOT_FOUND,
            "model_not_found",
            "requested model is not loaded by this server",
        );
    }

    let canceled = Arc::new(AtomicBool::new(false));
    let mut cancellation_guard = CancellationGuard {
        canceled: Arc::clone(&canceled),
        armed: true,
    };
    let (reply, receiver) = oneshot::channel();
    let request_id = state.request_sequence.fetch_add(1, Ordering::Relaxed);
    let job = Job {
        request: prepared,
        public_model: state.model.public_id.clone(),
        request_id,
        deadline,
        canceled,
        reply,
        _permit: permit,
    };
    match state.sender.try_send(WorkerCommand::Infer(job)) {
        Ok(()) => {}
        Err(TrySendError::Full(_)) => return overload(),
        Err(TrySendError::Disconnected(_)) => {
            state.ready.store(false, Ordering::Release);
            return api_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "worker_unavailable",
                "the resident inference worker is unavailable",
            );
        }
    }
    let remaining = match deadline.checked_duration_since(Instant::now()) {
        Some(remaining) if !remaining.is_zero() => remaining,
        _ => return timeout_response(),
    };
    let result = tokio::time::timeout(remaining, receiver).await;
    if result.is_ok() {
        cancellation_guard.armed = false;
    }
    match result {
        Err(_) => timeout_response(),
        Ok(Err(_)) => {
            state.ready.store(false, Ordering::Release);
            api_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "worker_unavailable",
                "the resident inference worker stopped before replying",
            )
        }
        Ok(Ok(Err(WorkerFailure::TimedOut))) => timeout_response(),
        Ok(Ok(Err(WorkerFailure::Canceled))) => api_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "request_canceled",
            "request was canceled before inference completed",
        ),
        Ok(Ok(Err(WorkerFailure::ShuttingDown))) => api_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "shutting_down",
            "server shutdown canceled queued inference",
        ),
        Ok(Ok(Err(WorkerFailure::UnsupportedShared(reason)))) => {
            tracing::warn!(reason = %reason, "required shared execution was rejected");
            api_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "unsupported_shared",
                "required shared execution cannot be satisfied for this request",
            )
        }
        Ok(Ok(Err(WorkerFailure::Inference(error)))) => {
            tracing::error!(code = %error.code, "resident inference request failed");
            let (status, error_type, message) = if is_terminal_inference_error(&error) {
                state.ready.store(false, Ordering::Release);
                (
                    StatusCode::SERVICE_UNAVAILABLE,
                    "worker_unavailable",
                    "the resident inference worker is unavailable",
                )
            } else if matches!(error.code.as_str(), "unsupported" | "validation") {
                (
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "validation_error",
                    "inference request is not supported by this server",
                )
            } else {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "inference_error",
                    "local inference failed; see server diagnostics",
                )
            };
            api_error(status, error_type, message)
        }
        Ok(Ok(Ok((response, disclosure)))) => success_response(response, disclosure),
    }
}

struct CancellationGuard {
    canceled: Arc<AtomicBool>,
    armed: bool,
}

impl Drop for CancellationGuard {
    fn drop(&mut self) {
        if self.armed {
            self.canceled.store(true, Ordering::Release);
        }
    }
}

fn is_terminal_inference_error(error: &CliError) -> bool {
    matches!(error.code.as_str(), "worker" | "backend_unavailable")
}

fn authorization_error(state: &AppState, headers: &HeaderMap) -> Option<Response> {
    let Some(secret) = &state.api_key else {
        return None;
    };
    let authorized = headers
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value == format!("Bearer {secret}"));
    if authorized {
        None
    } else {
        let mut response = api_error(
            StatusCode::UNAUTHORIZED,
            "authentication_error",
            "a valid bearer token is required",
        );
        response
            .headers_mut()
            .insert(header::WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
        Some(response)
    }
}

fn has_json_content_type(headers: &HeaderMap) -> bool {
    headers
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split(';').next())
        .is_some_and(|value| value.trim().eq_ignore_ascii_case("application/json"))
}

fn success_response(
    body: jev::SystemOneResponse,
    disclosure: jev::ExecutionDisclosure,
) -> Response {
    let mut response = json_response(StatusCode::OK, &body);
    insert_safe_header(
        response.headers_mut(),
        "x-openjev-execution",
        &format!(
            "requested={}; effective={}",
            disclosure.requested, disclosure.effective
        ),
    );
    if let Some(fallback) = disclosure.fallback {
        insert_safe_header(response.headers_mut(), "x-openjev-fallback", fallback);
    }
    insert_safe_header(
        response.headers_mut(),
        "x-openjev-probability-status",
        PROBABILITY_HEADER,
    );
    response
}

fn insert_safe_header(headers: &mut HeaderMap, name: &'static str, value: &str) {
    let bounded: String = value
        .chars()
        .filter(|character| character.is_ascii() && !character.is_ascii_control())
        .take(256)
        .collect();
    if let Ok(value) = HeaderValue::from_str(&bounded) {
        headers.insert(header::HeaderName::from_static(name), value);
    }
}

async fn method_not_allowed(method: Method) -> Response {
    api_error(
        StatusCode::METHOD_NOT_ALLOWED,
        "method_not_allowed",
        &format!("HTTP method {method} is not supported for this route"),
    )
}

async fn not_found() -> Response {
    api_error(
        StatusCode::NOT_FOUND,
        "not_found",
        "requested route was not found",
    )
}

fn overload() -> Response {
    let mut response = api_error(
        StatusCode::TOO_MANY_REQUESTS,
        "overloaded",
        "all resident inference admission slots are busy",
    );
    response.headers_mut().insert(
        header::RETRY_AFTER,
        HeaderValue::from_static(RETRY_AFTER_SECS),
    );
    response
}

fn timeout_response() -> Response {
    api_error(
        StatusCode::GATEWAY_TIMEOUT,
        "request_timeout",
        "the whole-request deadline elapsed",
    )
}

#[derive(Serialize)]
struct ApiErrorBody<'a> {
    error_type: &'a str,
    message: &'a str,
}

fn api_error(status: StatusCode, error_type: &str, message: &str) -> Response {
    json_response(
        status,
        &ApiErrorBody {
            error_type,
            message,
        },
    )
}

fn json_value(status: StatusCode, value: Value) -> Response {
    (status, Json(value)).into_response()
}

fn json_response<T: Serialize>(status: StatusCode, value: &T) -> Response {
    (status, Json(value)).into_response()
}

struct Job {
    request: jev::PreparedRequest,
    public_model: String,
    request_id: u64,
    deadline: Instant,
    canceled: Arc<AtomicBool>,
    reply: oneshot::Sender<JobResult>,
    _permit: tokio::sync::OwnedSemaphorePermit,
}

type JobResult = Result<(jev::SystemOneResponse, jev::ExecutionDisclosure), WorkerFailure>;

enum WorkerCommand {
    Infer(Job),
    Shutdown,
}

#[derive(Debug)]
enum WorkerFailure {
    TimedOut,
    Canceled,
    ShuttingDown,
    UnsupportedShared(String),
    Inference(CliError),
}

struct WorkerHandle {
    sender: SyncSender<WorkerCommand>,
    join: Option<JoinHandle<Result<(), CliError>>>,
    ready: Arc<AtomicBool>,
}

impl WorkerHandle {
    fn spawn<F>(
        factory: F,
        max_sequences: u32,
        require_shared: bool,
        shutdown: Arc<AtomicBool>,
    ) -> Result<(Self, StartupInfo), CliError>
    where
        F: FnOnce() -> Result<Box<dyn DecisionScorer>, CliError> + Send + 'static,
    {
        let (sender, receiver) = mpsc::sync_channel(MAX_ADMITTED_JOBS);
        let (startup_sender, startup_receiver) = mpsc::sync_channel(1);
        let ready = Arc::new(AtomicBool::new(false));
        let ready_for_thread = Arc::clone(&ready);
        let join = std::thread::Builder::new()
            .name("openjev-http-inference-owner".to_owned())
            .spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    worker_main(
                        factory,
                        receiver,
                        &startup_sender,
                        &ready_for_thread,
                        &shutdown,
                        max_sequences,
                        require_shared,
                    )
                }));
                ready_for_thread.store(false, Ordering::Release);
                match result {
                    Ok(result) => result,
                    Err(_) => {
                        tracing::error!("resident inference owner thread panicked");
                        Err(CliError::runtime(
                            "worker",
                            "resident inference owner thread panicked",
                        ))
                    }
                }
            })
            .map_err(|error| CliError::runtime("worker", error.to_string()))?;
        let startup = match startup_receiver.recv() {
            Ok(Ok(startup)) => startup,
            Ok(Err(error)) => {
                let _ = join_worker(join);
                return Err(error);
            }
            Err(_) => {
                return match join_worker(join) {
                    Err(error) => Err(error),
                    Ok(()) => Err(CliError::runtime(
                        "worker",
                        "resident inference owner stopped during startup",
                    )),
                };
            }
        };
        Ok((
            Self {
                sender,
                join: Some(join),
                ready,
            },
            startup,
        ))
    }

    fn sender(&self) -> SyncSender<WorkerCommand> {
        self.sender.clone()
    }

    fn ready(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.ready)
    }

    fn shutdown(mut self) -> Result<(), CliError> {
        self.ready.store(false, Ordering::Release);
        if let Some(join) = self.join.take() {
            let _ = self.sender.send(WorkerCommand::Shutdown);
            return join_worker(join);
        }
        Ok(())
    }
}

impl Drop for WorkerHandle {
    fn drop(&mut self) {
        self.ready.store(false, Ordering::Release);
        if let Some(join) = self.join.take() {
            let _ = self.sender.send(WorkerCommand::Shutdown);
            let _ = join.join();
        }
    }
}

fn join_worker(join: JoinHandle<Result<(), CliError>>) -> Result<(), CliError> {
    join.join()
        .map_err(|_| CliError::runtime("worker", "resident inference owner thread panicked"))?
}

fn worker_main<F>(
    factory: F,
    receiver: Receiver<WorkerCommand>,
    startup: &SyncSender<Result<StartupInfo, CliError>>,
    ready: &AtomicBool,
    shutdown: &AtomicBool,
    max_sequences: u32,
    require_shared: bool,
) -> Result<(), CliError>
where
    F: FnOnce() -> Result<Box<dyn DecisionScorer>, CliError>,
{
    let mut scorer = match factory() {
        Ok(scorer) => scorer,
        Err(error) => {
            let _ = startup.send(Err(error));
            return Ok(());
        }
    };
    let warmup = warmup(&mut *scorer);
    let info = match warmup {
        Ok(model) => startup_info(&model),
        Err(error) => {
            let shutdown_result = shutdown_scorer(&mut *scorer);
            let _ = startup.send(Err(error));
            return shutdown_result;
        }
    };
    ready.store(true, Ordering::Release);
    if startup.send(Ok(info)).is_err() {
        ready.store(false, Ordering::Release);
        return shutdown_scorer(&mut *scorer);
    }

    while let Ok(command) = receiver.recv() {
        match command {
            WorkerCommand::Shutdown => break,
            WorkerCommand::Infer(job) => {
                let Job {
                    request,
                    public_model,
                    request_id,
                    deadline,
                    canceled,
                    reply,
                    _permit,
                } = job;
                let result = if shutdown.load(Ordering::Acquire) {
                    Err(WorkerFailure::ShuttingDown)
                } else if canceled.load(Ordering::Acquire) || reply.is_closed() {
                    Err(WorkerFailure::Canceled)
                } else if Instant::now() >= deadline {
                    Err(WorkerFailure::TimedOut)
                } else {
                    score_job(
                        &mut *scorer,
                        request,
                        public_model,
                        request_id,
                        deadline,
                        &canceled,
                        shutdown,
                        max_sequences,
                        require_shared,
                    )
                };
                let terminal_failure = matches!(
                    &result,
                    Err(WorkerFailure::Inference(error)) if is_terminal_inference_error(error)
                );
                if terminal_failure {
                    ready.store(false, Ordering::Release);
                    shutdown.store(true, Ordering::Release);
                }
                let _ = reply.send(result);
                drop(_permit);
                if terminal_failure {
                    break;
                }
            }
        }
    }
    ready.store(false, Ordering::Release);
    shutdown_scorer(&mut *scorer)
}

fn shutdown_scorer(scorer: &mut dyn DecisionScorer) -> Result<(), CliError> {
    let result = scorer.shutdown();
    if let Err(error) = &result {
        tracing::error!(error = %error, "resident scorer shutdown failed");
    }
    result
}

fn warmup(scorer: &mut dyn DecisionScorer) -> Result<ModelMetadata, CliError> {
    let decision = Decision::new(
        "openjev-server-warmup",
        StateValue::string("openjev resident server warmup")
            .map_err(CliError::from_core_validation)?,
        "Select the first option.",
        vec![
            DecisionOption {
                id: "ready".to_owned(),
                description: "Ready".to_owned(),
            },
            DecisionOption {
                id: "not-ready".to_owned(),
                description: "Not ready".to_owned(),
            },
        ],
    )
    .map_err(CliError::from_core_validation)?;
    let readout = scorer.score_direct(decision)?;
    readout.validate().map_err(CliError::from_runtime_core)?;
    Ok(readout.model)
}

fn startup_info(model: &ModelMetadata) -> StartupInfo {
    let public_id = if model.source == "local" {
        format!("openjev-local-{}", &model.artifact_sha256[..12])
    } else {
        model.id.clone()
    };
    StartupInfo {
        description: format!(
            "Local OpenJev {} model served through {}",
            model.quant, model.backend
        ),
        public_id,
        // GGUF manifests do not carry a trustworthy publication date. The
        // API field remains explicit rather than inventing a TypeSafe release.
        release_date: "unknown",
    }
}

#[allow(clippy::too_many_arguments)]
fn score_job(
    scorer: &mut dyn DecisionScorer,
    mut request: jev::PreparedRequest,
    public_model: String,
    request_id: u64,
    deadline: Instant,
    canceled: &AtomicBool,
    shutdown: &AtomicBool,
    max_sequences: u32,
    require_shared: bool,
) -> JobResult {
    let inference = std::mem::take(&mut request.inference);
    if require_shared && inference.len() < 2 {
        return Err(WorkerFailure::UnsupportedShared(format!(
            "request has {} inferential questions; shared execution requires at least two",
            inference.len()
        )));
    }
    let mut readouts = Vec::with_capacity(inference.len());
    let group_id = format!("openjev-http-{request_id}");
    if inference.len() <= 1 {
        for item in &inference {
            check_job_state(deadline, canceled, shutdown)?;
            readouts.push(
                commands::score_item_with_reason(
                    scorer,
                    item,
                    ExecutionMode::Direct,
                    false,
                    None,
                    None,
                )
                .map_err(WorkerFailure::Inference)?,
            );
        }
    } else {
        let group_limit = max_sequences.saturating_sub(1).max(1) as usize;
        let mut native_failure = None;
        for group in inference.chunks(group_limit) {
            check_job_state(deadline, canceled, shutdown)?;
            match attempt_native_group(scorer, group, ExecutionMode::Shared, false, Some(&group_id))
            {
                Ok(mut rows) => readouts.append(&mut rows),
                Err(reason) => {
                    native_failure = Some(reason);
                    break;
                }
            }
        }
        if let Some(reason) = native_failure {
            if require_shared {
                return Err(WorkerFailure::UnsupportedShared(reason));
            }
            tracing::warn!(
                "resident request discarded tentative shared rows and is using fresh serial full-prompt fallback"
            );
            readouts.clear();
            for item in &inference {
                check_job_state(deadline, canceled, shutdown)?;
                readouts.push(
                    commands::score_item_with_reason(
                        scorer,
                        item,
                        ExecutionMode::Shared,
                        false,
                        Some(&group_id),
                        Some(&reason),
                    )
                    .map_err(WorkerFailure::Inference)?,
                );
            }
        }
    }
    check_job_state(deadline, canceled, shutdown)?;
    jev::project_response(request, readouts, public_model).map_err(WorkerFailure::Inference)
}

fn check_job_state(
    deadline: Instant,
    canceled: &AtomicBool,
    shutdown: &AtomicBool,
) -> Result<(), WorkerFailure> {
    if shutdown.load(Ordering::Acquire) {
        Err(WorkerFailure::ShuttingDown)
    } else if canceled.load(Ordering::Acquire) {
        Err(WorkerFailure::Canceled)
    } else if Instant::now() >= deadline {
        Err(WorkerFailure::TimedOut)
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::{
        convert::Infallible,
        sync::{Arc, Mutex},
    };

    use axum::{
        body::{Body, Bytes},
        http::Request as HttpRequest,
    };
    use futures_util::stream;
    use openjev_core::{
        Device, ExecutionMetadata, GpuLayersRequested, GpuLayersStatus, Integrity, SharedTiming,
        TemplateMetadataStatus, first_argmax, standard_limitations,
    };
    use serde_json::Value;
    use tower::ServiceExt as _;

    use super::*;

    #[derive(Default)]
    struct TestState {
        loads: usize,
        calls: usize,
        direct_calls: usize,
        shared_calls: usize,
        shutdowns: usize,
        seen_states: Vec<Value>,
    }

    struct TestScorer {
        state: Arc<Mutex<TestState>>,
        sleep: Duration,
    }

    struct PanicScorer(TestScorer);
    struct TerminalErrorScorer(TestScorer);
    struct SharedTestScorer(TestScorer);
    struct ShutdownErrorScorer(TestScorer);
    struct PanicShutdownScorer(TestScorer);

    impl DecisionScorer for PanicScorer {
        fn score_direct(&mut self, decision: Decision) -> Result<openjev_core::Readout, CliError> {
            if decision.id != "openjev-server-warmup" {
                panic!("injected worker failure");
            }
            self.0.score_direct(decision)
        }

        fn shutdown(&mut self) -> Result<(), CliError> {
            self.0.shutdown()
        }
    }

    impl DecisionScorer for TerminalErrorScorer {
        fn score_direct(&mut self, decision: Decision) -> Result<openjev_core::Readout, CliError> {
            if decision.id == "openjev-server-warmup" {
                return self.0.score_direct(decision);
            }
            let mut state = self.0.state.lock().unwrap();
            state.calls += 1;
            state.direct_calls += 1;
            drop(state);
            Err(CliError::runtime(
                "worker",
                "injected terminal engine worker failure",
            ))
        }

        fn shutdown(&mut self) -> Result<(), CliError> {
            self.0.shutdown()
        }
    }

    impl DecisionScorer for SharedTestScorer {
        fn score_direct(&mut self, decision: Decision) -> Result<openjev_core::Readout, CliError> {
            self.0.score_direct(decision)
        }

        fn probe_id(&self, mode: ExecutionMode) -> Result<String, String> {
            if mode == ExecutionMode::Shared {
                Ok("test-shared-probe".to_owned())
            } else {
                Err("only shared execution is supported by this test scorer".to_owned())
            }
        }

        fn score_shared(
            &mut self,
            decisions: Vec<Decision>,
            probe_id: String,
        ) -> Result<Vec<openjev_core::Readout>, CliError> {
            let mut state = self.0.state.lock().unwrap();
            state.calls += decisions.len();
            state.shared_calls += 1;
            state.seen_states.extend(
                decisions
                    .iter()
                    .map(|decision| decision.state.as_value().clone()),
            );
            drop(state);
            let batch_size = u64::try_from(decisions.len()).unwrap();
            let timing = SharedTiming {
                total_seconds: 0.0,
                encode_seconds: 0.0,
                prefix_tokens: 1,
                prefill_seconds: 0.0,
                replicate_seconds: 0.0,
                suffix_forward_seconds: 0.0,
                batch_size,
                true_suffix_tokens: batch_size,
                padded_suffix_tokens: batch_size,
            };
            Ok(decisions
                .into_iter()
                .map(|decision| {
                    let mut readout = test_readout(decision);
                    readout.execution.requested_mode = ExecutionMode::Shared;
                    readout.execution.effective_mode = ExecutionMode::Shared;
                    readout.execution.probe_id = Some(probe_id.clone());
                    readout.model.serving_config =
                        Some("llama-state-prefix-parallel-v1".to_owned());
                    readout.readout = "native selected suffix-position logits".to_owned();
                    readout.cache_hit = Some(true);
                    readout.prefix_tokens = Some(1);
                    readout.prefix_sha256 = Some("2".repeat(64));
                    readout.shared_timing = Some(timing.clone());
                    readout
                })
                .collect())
        }

        fn shutdown(&mut self) -> Result<(), CliError> {
            self.0.shutdown()
        }
    }

    impl DecisionScorer for ShutdownErrorScorer {
        fn score_direct(&mut self, decision: Decision) -> Result<openjev_core::Readout, CliError> {
            self.0.score_direct(decision)
        }

        fn shutdown(&mut self) -> Result<(), CliError> {
            self.0.shutdown()?;
            Err(CliError::runtime(
                "injected_shutdown",
                "injected scorer shutdown failure",
            ))
        }
    }

    impl DecisionScorer for PanicShutdownScorer {
        fn score_direct(&mut self, decision: Decision) -> Result<openjev_core::Readout, CliError> {
            self.0.score_direct(decision)
        }

        fn shutdown(&mut self) -> Result<(), CliError> {
            {
                self.0.state.lock().unwrap().shutdowns += 1;
            }
            panic!("injected scorer shutdown panic");
        }
    }

    impl DecisionScorer for TestScorer {
        fn score_direct(&mut self, decision: Decision) -> Result<openjev_core::Readout, CliError> {
            let warmup = decision.id == "openjev-server-warmup";
            if !warmup && !self.sleep.is_zero() {
                std::thread::sleep(self.sleep);
            }
            let mut state = self.state.lock().unwrap();
            state.calls += 1;
            state.direct_calls += 1;
            if !warmup {
                state.seen_states.push(decision.state.as_value().clone());
            }
            drop(state);
            Ok(test_readout(decision))
        }

        fn shutdown(&mut self) -> Result<(), CliError> {
            self.state.lock().unwrap().shutdowns += 1;
            Ok(())
        }
    }

    fn test_readout(decision: Decision) -> openjev_core::Readout {
        let count = decision.options.len();
        let probabilities = if count == 2 {
            vec![0.25, 0.75]
        } else {
            vec![1.0 / count as f64; count]
        };
        let option_ids: Vec<_> = decision
            .options
            .iter()
            .map(|option| option.id.clone())
            .collect();
        let choice_index = first_argmax(&probabilities).unwrap();
        let readout = openjev_core::Readout {
            schema: "openjev-readout-v1".to_owned(),
            id: decision.id,
            primitive: openjev_core::Primitive::Choice,
            choice: option_ids[choice_index].clone(),
            choice_index,
            option_ids,
            probabilities,
            option_logits: vec![0.0; count],
            answer_token_ids: (1..=u32::try_from(count).unwrap()).collect(),
            allowed_token_mass: 0.5,
            full_vocab_argmax_id: 1,
            full_vocab_log_normalizer: 1.0,
            input_tokens: 7,
            forward_seconds: None,
            total_seconds: None,
            prompt_sha256: "0".repeat(64),
            prompt_version: "direct-options-v1".to_owned(),
            model: ModelMetadata {
                id: "test-model".to_owned(),
                source: "test/repository".to_owned(),
                revision: "test-revision".to_owned(),
                file: "test.gguf".to_owned(),
                quant: "TEST".to_owned(),
                backend: "deterministic-test-backend".to_owned(),
                artifact_sha256: "1".repeat(64),
                integrity: Integrity::LocalUnverified,
                dtype: "test".to_owned(),
                native_reference: None,
                template_profile: openjev_core::PromptProfile::Qwen3,
                template_sha256: None,
                template_override: true,
                template_status: TemplateMetadataStatus::OverrideUnverified,
                template_equivalence_evidence: None,
                serving_config: Some("llama-direct-v1".to_owned()),
                adapter: None,
                adapter_sha256: None,
                adapter_revision: None,
                torch_version: None,
                transformers_version: None,
            },
            readout: openjev_core::DIRECT_READOUT.to_owned(),
            probability_status: openjev_core::PROBABILITY_STATUS.to_owned(),
            limitations: standard_limitations(),
            execution: ExecutionMetadata {
                requested_mode: ExecutionMode::Direct,
                effective_mode: ExecutionMode::Direct,
                fallback_reason: None,
                device: Device::Cpu,
                device_name: "test".to_owned(),
                gpu_layers_requested: GpuLayersRequested::Count(0),
                gpu_layers_actual: Some(0),
                gpu_layers_status: GpuLayersStatus::KnownDisabled,
                threads: 1,
                n_ctx_requested: None,
                n_ctx_actual: 128,
                max_tokens: 128,
                n_batch: 128,
                n_ubatch: 128,
                n_seq_max: 32,
                kv_unified: true,
                waves: 1,
                probe_id: None,
                run_id: "test-run".to_owned(),
                group_id: None,
            },
            confidence: None,
            confidence_status: None,
            p_yes: None,
            level_values: None,
            expected_value: None,
            argmax_level: None,
            cache_hit: Some(false),
            prefix_tokens: None,
            prefix_sha256: None,
            prefill_seconds: None,
            copy_seconds: None,
            suffix_forward_seconds: None,
            shared_timing: None,
            postprocess: None,
        };
        readout.validate().unwrap();
        readout
    }

    fn test_server(
        admission: usize,
        timeout: Duration,
        sleep: Duration,
        api_key: Option<&str>,
    ) -> (Router, WorkerHandle, Arc<Mutex<TestState>>) {
        test_server_mode(admission, timeout, sleep, api_key, false, false)
    }

    fn test_server_mode(
        admission: usize,
        timeout: Duration,
        sleep: Duration,
        api_key: Option<&str>,
        require_shared: bool,
        supports_shared: bool,
    ) -> (Router, WorkerHandle, Arc<Mutex<TestState>>) {
        let counters = Arc::new(Mutex::new(TestState::default()));
        let factory_state = Arc::clone(&counters);
        let shutdown = Arc::new(AtomicBool::new(false));
        let (worker, startup) = WorkerHandle::spawn(
            move || {
                factory_state.lock().unwrap().loads += 1;
                let scorer = TestScorer {
                    state: factory_state,
                    sleep,
                };
                if supports_shared {
                    Ok(Box::new(SharedTestScorer(scorer)) as Box<dyn DecisionScorer>)
                } else {
                    Ok(Box::new(scorer) as Box<dyn DecisionScorer>)
                }
            },
            32,
            require_shared,
            Arc::clone(&shutdown),
        )
        .unwrap();
        let app = router(AppState {
            sender: worker.sender(),
            admission: Arc::new(Semaphore::new(admission)),
            ready: worker.ready(),
            shutdown,
            request_timeout: timeout,
            api_key: api_key.map(Arc::<str>::from),
            model: startup,
            request_sequence: Arc::new(AtomicU64::new(1)),
        });
        (app, worker, counters)
    }

    async fn call(
        app: Router,
        method: Method,
        uri: &str,
        body: impl Into<Body>,
        content_type: Option<&str>,
        authorization: Option<&str>,
    ) -> (StatusCode, HeaderMap, Value) {
        let mut builder = HttpRequest::builder().method(method).uri(uri);
        if let Some(content_type) = content_type {
            builder = builder.header(header::CONTENT_TYPE, content_type);
        }
        if let Some(authorization) = authorization {
            builder = builder.header(header::AUTHORIZATION, authorization);
        }
        let response = app
            .oneshot(builder.body(body.into()).unwrap())
            .await
            .unwrap();
        let status = response.status();
        let headers = response.headers().clone();
        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let value = serde_json::from_slice(&bytes).unwrap();
        (status, headers, value)
    }

    fn request_body(state: &str) -> String {
        format!(
            r#"{{"state":{state},"questions":{{"route":{{"type":"choice","criteria":{{"a":null,"b":"B"}}}},"truth":{{"type":"noul"}},"score":{{"type":"score","criteria":["low","high"]}}}}}}"#
        )
    }

    #[test]
    fn loopback_and_nonloopback_auth_configuration_is_fail_closed() {
        assert!(ServeOptions::from_cli(None, None, None, None).is_ok());
        assert!(
            ServeOptions::from_cli(Some("0.0.0.0".parse().unwrap()), None, None, None).is_err()
        );
        assert!(ServeOptions::from_cli(None, Some(0), None, None).is_err());
        assert!(ServeOptions::from_cli(None, None, Some(0), None).is_err());
        assert!(ServeOptions::from_cli(None, None, Some(u64::MAX), None).is_err());
    }

    #[test]
    fn router_loads_once_serves_sdk_shapes_and_shuts_down_once() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let (app, worker, counters) = test_server(
                MAX_ADMITTED_JOBS,
                Duration::from_secs(2),
                Duration::ZERO,
                None,
            );
            let (status, _, health) = call(
                app.clone(),
                Method::GET,
                "/healthz",
                Body::empty(),
                None,
                None,
            )
            .await;
            assert_eq!(status, StatusCode::OK);
            assert_eq!(health["status"], "ok");
            let (status, _, ready) = call(
                app.clone(),
                Method::GET,
                "/readyz",
                Body::empty(),
                None,
                None,
            )
            .await;
            assert_eq!(status, StatusCode::OK);
            assert_eq!(ready["status"], "ready");
            let (status, _, models) = call(
                app.clone(),
                Method::GET,
                "/v1/models",
                Body::empty(),
                None,
                None,
            )
            .await;
            assert_eq!(status, StatusCode::OK);
            assert_eq!(models["models"][0]["name"], "test-model");
            assert_eq!(models["models"][0]["release_date"], "unknown");

            for state in [r#"{"request":1}"#, r#"{"request":2}"#] {
                let (status, headers, response) = call(
                    app.clone(),
                    Method::POST,
                    "/v1/systemone",
                    request_body(state),
                    Some("application/json; charset=utf-8"),
                    Some("Bearer dummy-sdk-token"),
                )
                .await;
                assert_eq!(status, StatusCode::OK, "{response:?}");
                assert_eq!(response["model"], "test-model");
                assert_eq!(response["answers"]["route"]["choice"], "b");
                assert_eq!(response["answers"]["truth"]["noul"], 0.25);
                assert_eq!(response["answers"]["score"]["score"], 0.75);
                assert_eq!(response["usage"]["input_tokens"], 21);
                assert_eq!(response["usage"]["output_tokens"], 0);
                assert!(headers.contains_key("x-openjev-execution"));
                assert!(headers.contains_key("x-openjev-fallback"));
                assert_eq!(headers["x-openjev-probability-status"], PROBABILITY_HEADER);
            }
            {
                let state = counters.lock().unwrap();
                assert_eq!(state.loads, 1);
                assert_eq!(state.calls, 7, "one warmup plus three calls per request");
                assert_eq!(state.seen_states.len(), 6);
                assert_ne!(state.seen_states[0], state.seen_states[3]);
            }
            worker.shutdown().unwrap();
            assert_eq!(counters.lock().unwrap().shutdowns, 1);
            let (status, _, ready) =
                call(app, Method::GET, "/readyz", Body::empty(), None, None).await;
            assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
            assert_eq!(ready["error_type"], "not_ready");
        });
    }

    #[test]
    fn require_shared_rejects_zero_or_one_inferential_question_and_preserves_other_modes() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let cases = [
                (
                    r#"{"state":"s","questions":{"only":{"type":"choice","criteria":{"only":null}}}}"#,
                    StatusCode::UNPROCESSABLE_ENTITY,
                    0,
                    0,
                ),
                (
                    r#"{"state":"s","questions":{"one":{"type":"noul"}}}"#,
                    StatusCode::UNPROCESSABLE_ENTITY,
                    0,
                    0,
                ),
                (
                    r#"{"state":"s","questions":{"one":{"type":"noul"},"two":{"type":"noul"}}}"#,
                    StatusCode::OK,
                    2,
                    1,
                ),
            ];
            for (body, expected_status, inferred_rows, shared_calls) in cases {
                let (app, worker, counters) = test_server_mode(
                    MAX_ADMITTED_JOBS,
                    Duration::from_secs(2),
                    Duration::ZERO,
                    None,
                    true,
                    true,
                );
                let baseline = counters.lock().unwrap().calls;
                let (status, _, response) = call(
                    app,
                    Method::POST,
                    "/v1/systemone",
                    body.to_owned(),
                    Some("application/json"),
                    None,
                )
                .await;
                assert_eq!(status, expected_status, "{response:?}");
                if status == StatusCode::UNPROCESSABLE_ENTITY {
                    assert_eq!(response["error_type"], "unsupported_shared");
                }
                {
                    let state = counters.lock().unwrap();
                    assert_eq!(state.calls, baseline + inferred_rows);
                    assert_eq!(state.direct_calls, 1, "only startup warmup may be direct");
                    assert_eq!(state.shared_calls, shared_calls);
                }
                worker.shutdown().unwrap();
            }

            let (app, worker, counters) = test_server_mode(
                MAX_ADMITTED_JOBS,
                Duration::from_secs(2),
                Duration::ZERO,
                None,
                true,
                false,
            );
            let baseline = counters.lock().unwrap().calls;
            let (status, _, response) = call(
                app,
                Method::POST,
                "/v1/systemone",
                r#"{"state":"s","questions":{"one":{"type":"noul"},"two":{"type":"noul"}}}"#
                    .to_owned(),
                Some("application/json"),
                None,
            )
            .await;
            assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{response:?}");
            assert_eq!(response["error_type"], "unsupported_shared");
            assert_eq!(
                counters.lock().unwrap().calls,
                baseline,
                "missing shared receipt must not fall back to direct inference"
            );
            worker.shutdown().unwrap();

            let (app, worker, counters) = test_server_mode(
                MAX_ADMITTED_JOBS,
                Duration::from_secs(2),
                Duration::ZERO,
                None,
                false,
                false,
            );
            let (status, _, response) = call(
                app,
                Method::POST,
                "/v1/systemone",
                r#"{"state":"s","questions":{"one":{"type":"noul"}}}"#.to_owned(),
                Some("application/json"),
                None,
            )
            .await;
            assert_eq!(status, StatusCode::OK, "{response:?}");
            {
                let state = counters.lock().unwrap();
                assert_eq!(state.calls, 2, "warmup plus one ordinary direct request");
                assert_eq!(state.direct_calls, 2);
                assert_eq!(state.shared_calls, 0);
            }
            worker.shutdown().unwrap();
        });
    }

    #[test]
    fn auth_validation_and_router_errors_are_json_and_do_not_infer() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let (app, worker, counters) = test_server(
                MAX_ADMITTED_JOBS,
                Duration::from_secs(2),
                Duration::ZERO,
                Some("secret"),
            );
            let baseline = counters.lock().unwrap().calls;
            let (status, headers, body) = call(
                app.clone(),
                Method::GET,
                "/v1/models",
                Body::empty(),
                None,
                None,
            )
            .await;
            assert_eq!(status, StatusCode::UNAUTHORIZED);
            assert_eq!(body["error_type"], "authentication_error");
            assert_eq!(headers[header::WWW_AUTHENTICATE], "Bearer");

            let cases = [
                (
                    Method::POST,
                    "/v1/systemone",
                    "{}",
                    None,
                    StatusCode::UNAUTHORIZED,
                ),
                (
                    Method::POST,
                    "/v1/systemone",
                    "{}",
                    Some("text/plain"),
                    StatusCode::UNAUTHORIZED,
                ),
                (Method::GET, "/missing", "", None, StatusCode::NOT_FOUND),
                (
                    Method::GET,
                    "/v1/systemone",
                    "",
                    None,
                    StatusCode::METHOD_NOT_ALLOWED,
                ),
            ];
            for (method, uri, body_text, content_type, expected) in cases {
                let (status, _, body) = call(
                    app.clone(),
                    method,
                    uri,
                    body_text.to_owned(),
                    content_type,
                    None,
                )
                .await;
                assert_eq!(status, expected);
                assert!(body["error_type"].is_string());
                assert!(body["message"].is_string());
            }
            let (status, _, _) = call(
                app.clone(),
                Method::POST,
                "/v1/systemone",
                "{bad}".to_owned(),
                Some("application/json"),
                Some("Bearer secret"),
            )
            .await;
            assert_eq!(status, StatusCode::BAD_REQUEST);
            let (status, _, _) = call(
                app.clone(),
                Method::POST,
                "/v1/systemone",
                request_body("1.5"),
                Some("application/json"),
                Some("Bearer secret"),
            )
            .await;
            assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
            let (status, _, _) = call(
                app.clone(),
                Method::POST,
                "/v1/systemone",
                request_body("\"s\""),
                Some("text/plain"),
                Some("Bearer secret"),
            )
            .await;
            assert_eq!(status, StatusCode::UNSUPPORTED_MEDIA_TYPE);
            let large_state = "x".repeat(70_000);
            let many_questions = (0..64)
                .map(|index| format!("\"q{index}\":{{\"type\":\"noul\"}}"))
                .collect::<Vec<_>>()
                .join(",");
            let expanded_body =
                format!("{{\"state\":\"{large_state}\",\"questions\":{{{many_questions}}}}}");
            assert!(expanded_body.len() < MAX_BODY_BYTES);
            let (status, _, body) = call(
                app.clone(),
                Method::POST,
                "/v1/systemone",
                expanded_body,
                Some("application/json"),
                Some("Bearer secret"),
            )
            .await;
            assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
            assert_eq!(body["error_type"], "validation_error");
            assert_eq!(counters.lock().unwrap().calls, baseline);
            let chunks = stream::iter([
                Ok::<_, Infallible>(Bytes::from(vec![b' '; MAX_BODY_BYTES / 2 + 1])),
                Ok(Bytes::from(vec![b' '; MAX_BODY_BYTES / 2 + 1])),
            ]);
            let (status, _, body) = call(
                app,
                Method::POST,
                "/v1/systemone",
                Body::from_stream(chunks),
                Some("application/json"),
                Some("Bearer secret"),
            )
            .await;
            assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
            assert_eq!(body["error_type"], "body_too_large");
            assert_eq!(counters.lock().unwrap().calls, baseline);
            worker.shutdown().unwrap();
        });
    }

    #[test]
    fn deadline_cancels_queued_work_and_inflight_work_keeps_admission_charged() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let (app, worker, counters) = test_server(
                1,
                Duration::from_millis(30),
                Duration::from_millis(120),
                None,
            );
            let first_app = app.clone();
            let first = tokio::spawn(async move {
                call(
                    first_app,
                    Method::POST,
                    "/v1/systemone",
                    request_body("\"first\""),
                    Some("application/json"),
                    None,
                )
                .await
            });
            tokio::time::sleep(Duration::from_millis(10)).await;
            let (status, headers, body) = call(
                app.clone(),
                Method::POST,
                "/v1/systemone",
                request_body("\"second\""),
                Some("application/json"),
                None,
            )
            .await;
            assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
            assert_eq!(headers[header::RETRY_AFTER], RETRY_AFTER_SECS);
            assert_eq!(body["error_type"], "overloaded");
            let (status, _, body) = first.await.unwrap();
            assert_eq!(status, StatusCode::GATEWAY_TIMEOUT);
            assert_eq!(body["error_type"], "request_timeout");
            tokio::time::sleep(Duration::from_millis(120)).await;
            assert_eq!(
                counters.lock().unwrap().calls,
                2,
                "timed-out native work finishes, but no overloaded request is scored"
            );
            worker.shutdown().unwrap();
        });
    }

    #[test]
    fn worker_failure_clears_readiness_and_does_not_hang_reply_receivers() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let counters = Arc::new(Mutex::new(TestState::default()));
            let factory_state = Arc::clone(&counters);
            let shutdown = Arc::new(AtomicBool::new(false));
            let (worker, startup) = WorkerHandle::spawn(
                move || {
                    factory_state.lock().unwrap().loads += 1;
                    Ok(Box::new(PanicScorer(TestScorer {
                        state: factory_state,
                        sleep: Duration::ZERO,
                    })) as Box<dyn DecisionScorer>)
                },
                32,
                false,
                Arc::clone(&shutdown),
            )
            .unwrap();
            let app = router(AppState {
                sender: worker.sender(),
                admission: Arc::new(Semaphore::new(1)),
                ready: worker.ready(),
                shutdown,
                request_timeout: Duration::from_secs(1),
                api_key: None,
                model: startup,
                request_sequence: Arc::new(AtomicU64::new(1)),
            });
            let (status, _, body) = call(
                app.clone(),
                Method::POST,
                "/v1/systemone",
                request_body("\"panic\""),
                Some("application/json"),
                None,
            )
            .await;
            assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
            assert_eq!(body["error_type"], "worker_unavailable");
            let (status, _, _) = call(app, Method::GET, "/readyz", Body::empty(), None, None).await;
            assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
            let error = worker.shutdown().unwrap_err();
            assert_eq!(error.code, "worker");
        });
    }

    #[test]
    fn terminal_engine_worker_error_clears_readiness_and_stops_later_inference() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let counters = Arc::new(Mutex::new(TestState::default()));
            let factory_state = Arc::clone(&counters);
            let shutdown = Arc::new(AtomicBool::new(false));
            let (worker, startup) = WorkerHandle::spawn(
                move || {
                    Ok(Box::new(TerminalErrorScorer(TestScorer {
                        state: factory_state,
                        sleep: Duration::ZERO,
                    })) as Box<dyn DecisionScorer>)
                },
                32,
                false,
                Arc::clone(&shutdown),
            )
            .unwrap();
            let app = router(AppState {
                sender: worker.sender(),
                admission: Arc::new(Semaphore::new(2)),
                ready: worker.ready(),
                shutdown,
                request_timeout: Duration::from_secs(1),
                api_key: None,
                model: startup,
                request_sequence: Arc::new(AtomicU64::new(1)),
            });
            let (status, _, body) = call(
                app.clone(),
                Method::POST,
                "/v1/systemone",
                r#"{"state":"s","questions":{"one":{"type":"noul"}}}"#.to_owned(),
                Some("application/json"),
                None,
            )
            .await;
            assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
            assert_eq!(body["error_type"], "worker_unavailable");
            let (status, _, body) = call(
                app.clone(),
                Method::GET,
                "/readyz",
                Body::empty(),
                None,
                None,
            )
            .await;
            assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
            assert_eq!(body["error_type"], "not_ready");
            let (status, _, _) = call(
                app,
                Method::POST,
                "/v1/systemone",
                r#"{"state":"s","questions":{"later":{"type":"noul"}}}"#.to_owned(),
                Some("application/json"),
                None,
            )
            .await;
            assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
            assert_eq!(
                counters.lock().unwrap().calls,
                2,
                "warmup plus first request"
            );
            worker.shutdown().unwrap();
            assert_eq!(counters.lock().unwrap().shutdowns, 1);
        });
    }

    #[test]
    fn worker_shutdown_error_and_panic_are_reported_once() {
        for panic_on_shutdown in [false, true] {
            let counters = Arc::new(Mutex::new(TestState::default()));
            let factory_state = Arc::clone(&counters);
            let shutdown = Arc::new(AtomicBool::new(false));
            let (worker, _) = WorkerHandle::spawn(
                move || {
                    let scorer = TestScorer {
                        state: factory_state,
                        sleep: Duration::ZERO,
                    };
                    if panic_on_shutdown {
                        Ok(Box::new(PanicShutdownScorer(scorer)) as Box<dyn DecisionScorer>)
                    } else {
                        Ok(Box::new(ShutdownErrorScorer(scorer)) as Box<dyn DecisionScorer>)
                    }
                },
                32,
                false,
                Arc::clone(&shutdown),
            )
            .unwrap();
            let error = worker.shutdown().unwrap_err();
            assert_eq!(
                error.code,
                if panic_on_shutdown {
                    "worker"
                } else {
                    "injected_shutdown"
                }
            );
            assert_eq!(counters.lock().unwrap().shutdowns, 1);
        }
    }

    #[test]
    fn disconnected_queued_request_is_dropped_before_inference() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let (app, worker, counters) =
                test_server(2, Duration::from_secs(1), Duration::from_millis(100), None);
            let body = |state: &str| {
                format!(r#"{{"state":"{state}","questions":{{"q":{{"type":"noul"}}}}}}"#)
            };
            let first = tokio::spawn(call(
                app.clone(),
                Method::POST,
                "/v1/systemone",
                body("first"),
                Some("application/json"),
                None,
            ));
            tokio::time::sleep(Duration::from_millis(5)).await;
            let queued = tokio::spawn(call(
                app,
                Method::POST,
                "/v1/systemone",
                body("disconnected"),
                Some("application/json"),
                None,
            ));
            tokio::time::sleep(Duration::from_millis(10)).await;
            queued.abort();
            assert_eq!(first.await.unwrap().0, StatusCode::OK);
            tokio::time::sleep(Duration::from_millis(50)).await;
            assert_eq!(
                counters.lock().unwrap().calls,
                2,
                "one warmup plus the first request; disconnected queued work is skipped"
            );
            worker.shutdown().unwrap();
        });
    }

    #[test]
    fn queued_timeout_is_dropped_before_inference() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let (app, worker, counters) = test_server(
                2,
                Duration::from_millis(30),
                Duration::from_millis(100),
                None,
            );
            let first = tokio::spawn(call(
                app.clone(),
                Method::POST,
                "/v1/systemone",
                request_body("\"first\""),
                Some("application/json"),
                None,
            ));
            tokio::time::sleep(Duration::from_millis(5)).await;
            let second = tokio::spawn(call(
                app,
                Method::POST,
                "/v1/systemone",
                request_body("\"queued\""),
                Some("application/json"),
                None,
            ));
            assert_eq!(first.await.unwrap().0, StatusCode::GATEWAY_TIMEOUT);
            assert_eq!(second.await.unwrap().0, StatusCode::GATEWAY_TIMEOUT);
            tokio::time::sleep(Duration::from_millis(100)).await;
            assert_eq!(
                counters.lock().unwrap().calls,
                2,
                "one warmup plus only the first noninterruptible inference"
            );
            worker.shutdown().unwrap();
        });
    }
}
