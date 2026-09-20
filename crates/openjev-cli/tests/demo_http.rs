use std::{
    net::TcpListener,
    process::{Command, Output},
    sync::{Arc, Mutex},
    thread::JoinHandle,
    time::Duration,
};

use axum::{
    Json, Router,
    http::{HeaderMap, StatusCode},
    routing::post,
};
use serde_json::{Value, json};
use tokio::sync::oneshot;

type Requests = Arc<Mutex<Vec<(HeaderMap, Value)>>>;

struct Server {
    url: String,
    requests: Requests,
    shutdown: Option<oneshot::Sender<()>>,
    thread: Option<JoinHandle<()>>,
}

impl Server {
    fn new(status: StatusCode, body: String, delay: Duration) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        listener.set_nonblocking(true).unwrap();
        let requests = Requests::default();
        let recorded = requests.clone();
        let (shutdown, receiver) = oneshot::channel();
        let thread = std::thread::spawn(move || {
            tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap().block_on(async move {
                let app = Router::new().route("/v1/systemone", post(move |headers: HeaderMap, Json(request): Json<Value>| {
                    let recorded = recorded.clone();
                    let body = body.clone();
                    async move {
                        recorded.lock().unwrap().push((headers, request));
                        tokio::time::sleep(delay).await;
                        (status, [
                            ("x-openjev-execution", "requested=shared; effective=serial"),
                            ("x-openjev-fallback", "shared unavailable; serial full-prompt fallback"),
                            ("x-openjev-probability-status", "conditional option score; uncalibrated as decision confidence"),
                            ("location", "http://127.0.0.1:1/never-follow"),
                        ], body)
                    }
                }));
                axum::serve(tokio::net::TcpListener::from_std(listener).unwrap(), app)
                    .with_graceful_shutdown(async { let _ = receiver.await; }).await.unwrap();
            });
        });
        Self {
            url,
            requests,
            shutdown: Some(shutdown),
            thread: Some(thread),
        }
    }

    fn run(&self, args: &[&str]) -> Output {
        Command::new(env!("CARGO_BIN_EXE_openjev"))
            .args(["demo", "--base-url", &format!("{}/v1", self.url)])
            .args(args)
            .env("OPENJEV_DEMO_TEST_KEY", "demo-only-test-secret")
            .env_remove("OPENJEV_DEMO_MISSING_KEY")
            .output()
            .unwrap()
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.shutdown.take().unwrap().send(());
        self.thread.take().unwrap().join().unwrap();
    }
}

fn response() -> String {
    json!({"model": "mock-resident", "answers": {"result": {"type": "noul", "noul": 0.75}}, "usage": {"input_tokens": 42, "output_tokens": 0}}).to_string()
}

#[test]
fn eight_real_posts_use_auth_and_model_and_preserve_response_and_disclosures() {
    let server = Server::new(StatusCode::OK, response(), Duration::ZERO);
    let output = server.run(&[
        "--api-key-env",
        "OPENJEV_DEMO_TEST_KEY",
        "--model",
        "custom-resident",
    ]);
    assert!(output.status.success(), "{output:?}");
    let text = String::from_utf8(output.stdout).unwrap();
    let rows: Vec<Value> = text
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(rows.len(), 8);
    for row in &rows {
        assert!(row["example"].is_string());
        assert!(row["elapsed_ms"].as_f64().unwrap() >= 0.0);
        assert_eq!(
            row["response"],
            serde_json::from_str::<Value>(&response()).unwrap()
        );
        assert_eq!(
            row["metadata"]["x-openjev-execution"],
            "requested=shared; effective=serial"
        );
        assert!(row["metadata"]["x-openjev-fallback"].is_string());
        assert!(row["metadata"]["x-openjev-probability-status"].is_string());
    }
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("[8/8] Mixed ticket triage"));
    assert!(stderr.contains("not accuracy tests"));
    assert!(!stderr.contains("demo-only-test-secret"));
    let requests = server.requests.lock().unwrap();
    assert_eq!(requests.len(), 8);
    let mut types = std::collections::BTreeSet::new();
    for (row, (headers, request)) in rows.iter().zip(requests.iter()) {
        assert_eq!(&row["request"], request);
        assert_eq!(headers["authorization"], "Bearer demo-only-test-secret");
        assert_eq!(headers["content-type"], "application/json");
        assert_eq!(request["model"], "custom-resident");
        openjev_cli::server::jev::parse_request(&serde_json::to_vec(request).unwrap()).unwrap();
        for question in request["questions"].as_object().unwrap().values() {
            types.insert(question["type"].as_str().unwrap());
        }
    }
    assert_eq!(
        types.into_iter().collect::<Vec<_>>(),
        ["choice", "noul", "score"]
    );
}

#[test]
fn pretty_is_one_json_array_and_quiet_suppresses_progress_not_metadata() {
    let server = Server::new(StatusCode::OK, response(), Duration::ZERO);
    let output = server.run(&["--pretty", "--quiet"]);
    assert!(output.status.success(), "{output:?}");
    assert!(output.stderr.is_empty());
    let rows: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(rows.as_array().unwrap().len(), 8);
    assert!(rows[7]["metadata"]["x-openjev-fallback"].is_string());
    for (headers, request) in server.requests.lock().unwrap().iter() {
        assert!(!headers.contains_key("authorization"));
        assert_eq!(request["model"], "jev-latest");
    }
}

#[test]
fn http_errors_redirects_and_bad_bodies_stop_without_retry_or_stdout_noise() {
    for (status, body, code) in [
        (StatusCode::UNAUTHORIZED, "secret-body".into(), "demo_http"),
        (
            StatusCode::TOO_MANY_REQUESTS,
            "secret-body".into(),
            "demo_http",
        ),
        (StatusCode::FOUND, "secret-body".into(), "demo_http"),
        (StatusCode::OK, "not JSON".into(), "demo_response"),
        (StatusCode::OK, "{}".into(), "demo_response"),
        (StatusCode::OK, " ".repeat(1024 * 1024 + 1), "demo_response"),
    ] {
        let server = Server::new(status, body, Duration::ZERO);
        let output = server.run(&["--quiet"]);
        assert_eq!(output.status.code(), Some(1), "{output:?}");
        assert!(output.stdout.is_empty());
        let error: Value = serde_json::from_slice(&output.stderr).unwrap();
        assert_eq!(error["error"]["code"], code);
        assert!(!String::from_utf8_lossy(&output.stderr).contains("secret-body"));
        assert_eq!(server.requests.lock().unwrap().len(), 1);
    }
}

#[test]
fn timeout_and_missing_auth_env_are_actionable_failures() {
    let server = Server::new(StatusCode::OK, response(), Duration::from_secs(2));
    let output = server.run(&["--quiet", "--timeout-secs", "1"]);
    assert_eq!(output.status.code(), Some(1), "{output:?}");
    assert!(output.stdout.is_empty());
    let error: Value = serde_json::from_slice(&output.stderr).unwrap();
    assert!(
        error["error"]["message"]
            .as_str()
            .unwrap()
            .contains("timed out")
    );
    let output = server.run(&["--quiet", "--api-key-env", "OPENJEV_DEMO_MISSING_KEY"]);
    assert_eq!(output.status.code(), Some(2), "{output:?}");
    assert!(output.stdout.is_empty());
    assert_eq!(server.requests.lock().unwrap().len(), 1);
}

#[test]
fn unavailable_server_suggests_starting_serve() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    drop(listener);
    let output = Command::new(env!("CARGO_BIN_EXE_openjev"))
        .args(["demo", "--base-url", &url, "--quiet"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1), "{output:?}");
    assert!(output.stdout.is_empty());
    let error: Value = serde_json::from_slice(&output.stderr).unwrap();
    assert!(
        error["error"]["message"]
            .as_str()
            .unwrap()
            .contains("openjev serve")
    );
}
