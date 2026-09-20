#![cfg(all(feature = "native", feature = "integration"))]

use std::{
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    process::{Child, Command, Stdio},
    thread,
    time::{Duration, Instant},
};

use serde_json::Value;

#[cfg(all(feature = "metal", target_os = "macos"))]
const TEST_DEVICE: &str = "metal";
#[cfg(not(all(feature = "metal", target_os = "macos")))]
const TEST_DEVICE: &str = "cpu";
#[cfg(all(feature = "metal", target_os = "macos"))]
const REQUEST_TIMEOUT_SECS: u64 = 40;
#[cfg(not(all(feature = "metal", target_os = "macos")))]
const REQUEST_TIMEOUT_SECS: u64 = 120;
#[cfg(all(feature = "metal", target_os = "macos"))]
const READINESS_TIMEOUT_SECS: u64 = 90;
#[cfg(not(all(feature = "metal", target_os = "macos")))]
const READINESS_TIMEOUT_SECS: u64 = 180;

struct ChildGuard(Child);

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn http(port: u16, method: &str, path: &str, body: Option<&str>) -> (u16, String, Value) {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(REQUEST_TIMEOUT_SECS + 15)))
        .unwrap();
    let body = body.unwrap_or("");
    write!(
        stream,
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer local-integration\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    stream.flush().unwrap();
    let mut response = String::new();
    stream.read_to_string(&mut response).unwrap();
    let (headers, body) = response.split_once("\r\n\r\n").unwrap();
    let status = headers
        .lines()
        .next()
        .unwrap()
        .split_whitespace()
        .nth(1)
        .unwrap()
        .parse()
        .unwrap();
    (
        status,
        headers.to_owned(),
        serde_json::from_str(body).unwrap(),
    )
}

#[test]
fn cached_qwen_resident_server_reuses_one_process_and_stops_on_sigterm() {
    if std::env::var("OPENJEV_INTEGRATION").as_deref() != Ok("1") {
        return;
    }
    let port = {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        listener.local_addr().unwrap().port()
    };
    let child = Command::new(env!("CARGO_BIN_EXE_openjev"))
        .args([
            "--serve",
            "--offline",
            "--model",
            "qwen3-0.6b",
            "--device",
            TEST_DEVICE,
            "--host",
            "127.0.0.1",
            "--port",
            &port.to_string(),
            "--request-timeout-secs",
            &REQUEST_TIMEOUT_SECS.to_string(),
            "--quiet",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .unwrap();
    let mut child = ChildGuard(child);

    let deadline = Instant::now() + Duration::from_secs(READINESS_TIMEOUT_SECS);
    loop {
        if let Some(status) = child.0.try_wait().unwrap() {
            panic!("server exited before readiness ({status}); see inherited stderr");
        }
        if TcpStream::connect(("127.0.0.1", port)).is_ok() {
            let (status, _, body) = http(port, "GET", "/readyz", None);
            if status == 200 && body["status"] == "ready" {
                break;
            }
        }
        assert!(Instant::now() < deadline, "server readiness timed out");
        thread::sleep(Duration::from_millis(100));
    }

    let request = serde_json::json!({
        "model":"jev-latest",
        "state":{"ticket":"duplicate charge","severity":3},
        "questions":{
            "route":{"type":"choice","criteria":{"billing":"payments","support":"general"}},
            "review":{"type":"noul","instructions":"Does a human need to review this?"},
            "urgency":{"type":"score","criteria":["low","medium","high"]}
        }
    })
    .to_string();
    let first = http(port, "POST", "/v1/systemone", Some(&request));
    let second = http(port, "POST", "/v1/systemone", Some(&request));
    assert_eq!(first.0, 200, "{:?}", first.2);
    assert_eq!(second.0, 200, "{:?}", second.2);
    assert_eq!(first.2, second.2);
    assert_eq!(first.2["model"], "qwen3-0.6b");
    assert_eq!(first.2["answers"]["route"]["type"], "choice");
    assert_eq!(first.2["answers"]["review"]["type"], "noul");
    assert_eq!(first.2["answers"]["urgency"]["type"], "score");
    assert!(first.2["usage"]["input_tokens"].as_u64().unwrap() > 0);
    assert_eq!(first.2["usage"]["output_tokens"], 0);
    assert!(first.1.to_ascii_lowercase().contains("x-openjev-fallback:"));
    assert!(child.0.try_wait().unwrap().is_none());

    #[cfg(unix)]
    {
        let status = Command::new("/bin/kill")
            .args(["-TERM", &child.0.id().to_string()])
            .status()
            .unwrap();
        assert!(status.success());
    }
    #[cfg(not(unix))]
    child.0.kill().unwrap();

    let deadline = Instant::now() + Duration::from_secs(45);
    let status = loop {
        if let Some(status) = child.0.try_wait().unwrap() {
            break status;
        }
        assert!(
            Instant::now() < deadline,
            "server SIGTERM shutdown timed out"
        );
        thread::sleep(Duration::from_millis(50));
    };
    assert!(status.success(), "server shutdown status: {status}");
    let mut stdout = String::new();
    child
        .0
        .stdout
        .take()
        .unwrap()
        .read_to_string(&mut stdout)
        .unwrap();
    assert!(
        stdout.is_empty(),
        "server wrote human output to stdout: {stdout:?}"
    );
}
