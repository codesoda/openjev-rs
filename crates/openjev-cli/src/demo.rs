//! HTTP-only examples: never initialize a backend or download model weights.
use std::{
    io::Write,
    time::{Duration, Instant},
};

use reqwest::{Client, Url, header::HeaderValue};
use serde_json::{Value, json};

use crate::{
    CliError,
    args::{DemoArgs, GlobalArgs},
    output,
};

const MAX_RESPONSE_BYTES: usize = 1024 * 1024;

struct Example {
    name: &'static str,
    request: Value,
}

fn examples(model: &str) -> Vec<Example> {
    let fixtures = [
        (
            "Support routing",
            json!({"ticket": "I was charged twice for my subscription."}),
            json!({
                "route": {"type": "choice", "instructions": "Which team should handle this ticket?", "criteria": {"billing": "Payments and invoices", "technical": "Technical support", "sales": "New purchases"}}
            }),
        ),
        (
            "Tool selection",
            json!(
                "The user wants the latest weather forecast for Paris. No weather data is in the conversation."
            ),
            json!({
                "tool": {"type": "choice", "instructions": "Which tool should the assistant use next?", "criteria": {"search": "Search the web for current information", "calculator": "Calculate a numeric expression", "none": "Answer from the existing conversation"}}
            }),
        ),
        (
            "Message intent",
            json!("Please cancel my subscription at the end of this billing period."),
            json!({
                "intent": {"type": "choice", "instructions": "What does the customer want?", "criteria": {"cancel": "Cancel the subscription", "upgrade": "Upgrade the subscription", "refund": "Refund a past payment"}}
            }),
        ),
        (
            "Explicit refund request",
            json!("The parcel never arrived. Please refund my payment."),
            json!({
                "refund": {"type": "noul", "instructions": "Does the customer explicitly request a refund?"}
            }),
        ),
        (
            "Missing information",
            json!("Book me a flight tomorrow. The user has not supplied an origin or destination."),
            json!({
                "clarify": {"type": "noul", "instructions": "Must the assistant ask for missing information before booking?"}
            }),
        ),
        (
            "Incident urgency",
            json!({"incident": "Checkout is down for every customer. No purchases can complete.", "affected_percent": 100}),
            json!({
                "urgency": {"type": "score", "instructions": "How urgent is this incident?", "criteria": ["low: cosmetic issue", "medium: partial degradation", "high: critical service unavailable"]}
            }),
        ),
        (
            "Evidence sufficiency",
            json!(
                "The question asks whether the test suite passes. The agent ran the full suite on the current commit: 84 passed, 0 failed, exit code 0."
            ),
            json!({
                "evidence": {"type": "score", "instructions": "How strong is the evidence that the tests pass?", "criteria": ["none", "weak", "strong"]}
            }),
        ),
        (
            "Mixed ticket triage",
            json!({"ticket": "Our production API is returning 500 errors for all requests. Please have an engineer investigate immediately."}),
            json!({
                "route": {"type": "choice", "instructions": "Which team should handle this?", "criteria": {"engineering": "Service outages and defects", "billing": "Payments and invoices", "sales": "Plans and purchases"}},
                "review": {"type": "noul", "instructions": "Does this ticket explicitly request human investigation?"},
                "urgency": {"type": "score", "instructions": "How urgent is this ticket?", "criteria": ["low", "medium", "high"]}
            }),
        ),
    ];
    fixtures
        .into_iter()
        .map(|(name, state, questions)| Example {
            name,
            request: json!({"model": model, "state": state, "questions": questions}),
        })
        .collect()
}

fn endpoint(base: &str) -> Result<Url, CliError> {
    let mut url = Url::parse(base).map_err(|_| {
        CliError::validation(
            "--base-url must be an HTTP(S) server origin, optionally ending in /v1",
        )
    })?;
    if !matches!(url.scheme(), "http" | "https")
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
        || !matches!(url.path(), "" | "/" | "/v1" | "/v1/")
    {
        return Err(CliError::validation(
            "--base-url must be an HTTP(S) server origin, optionally ending in /v1; credentials, query strings, and fragments are not allowed",
        ));
    }
    url.set_path("/v1/systemone");
    Ok(url)
}

fn transport_error(error: reqwest::Error) -> CliError {
    // Do not echo credentials, response bodies, or untrusted URLs in diagnostics.
    let message = if error.is_connect() {
        "Cannot connect to the demo server. Start `openjev serve` in another terminal, wait for it to be ready, and check --base-url."
    } else if error.is_timeout() {
        "Demo request timed out. Check server load or increase --timeout-secs."
    } else {
        "Demo HTTP request failed. Check the server and --base-url."
    };
    CliError::runtime("demo_http", message)
}

fn io_error(error: std::io::Error) -> CliError {
    CliError::runtime("io", error.to_string())
}

pub(crate) fn execute<W: Write, E: Write>(
    global: &GlobalArgs,
    args: DemoArgs,
    stdout: &mut W,
    stderr: &mut E,
) -> Result<i32, CliError> {
    let url = endpoint(&args.base_url)?;
    let bearer = args.api_key_env.as_deref().map(|name| {
        let secret = std::env::var(name).map_err(|_| CliError::validation("the environment variable named by --api-key-env must contain a nonempty bearer secret"))?;
        if secret.trim().is_empty() {
            return Err(CliError::validation("the environment variable named by --api-key-env must contain a nonempty bearer secret"));
        }
        let mut header = HeaderValue::from_str(&format!("Bearer {secret}"))
            .map_err(|_| CliError::validation("invalid bearer secret in --api-key-env"))?;
        header.set_sensitive(true);
        Ok(header)
    }).transpose()?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| CliError::runtime("demo_http", error.to_string()))?;
    runtime.block_on(async {
        let client = Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .no_proxy()
            .connect_timeout(Duration::from_secs(5))
            .timeout(Duration::from_secs(args.timeout_secs))
            .build().map_err(transport_error)?;
        let examples = examples(global.model.as_deref().unwrap_or("jev-latest"));
        let mut results = Vec::new();
        if !global.quiet {
            writeln!(stderr, "Running {} examples against {url}; no model is loaded by this client.\nThese are demonstrations, not accuracy tests. Probabilities are conditional and uncalibrated.", examples.len()).map_err(io_error)?;
        }
        for (index, example) in examples.iter().enumerate() {
            if !global.quiet {
                writeln!(stderr, "\n[{}/{}] {}\nState: {}\nQuestions: {}", index + 1, examples.len(), example.name, example.request["state"], example.request["questions"]).map_err(io_error)?;
            }
            let started = Instant::now();
            let mut request = client.post(url.clone()).json(&example.request);
            if let Some(bearer) = &bearer {
                request = request.header(reqwest::header::AUTHORIZATION, bearer.clone());
            }
            let mut response = request.send().await.map_err(transport_error)?;
            if !response.status().is_success() {
                return Err(CliError::runtime("demo_http", format!("Demo example {} returned HTTP {}. Check server logs, --model, and --api-key-env; stopping without retrying.", index + 1, response.status().as_u16())));
            }
            let metadata: serde_json::Map<String, Value> = [
                "x-openjev-execution", "x-openjev-fallback", "x-openjev-probability-status",
            ].into_iter().filter_map(|name| {
                response.headers().get(name).and_then(|value| value.to_str().ok())
                    .map(|value| (name.to_owned(), json!(value)))
            }).collect();
            let mut bytes = Vec::new();
            while let Some(chunk) = response.chunk().await.map_err(transport_error)? {
                if bytes.len() + chunk.len() > MAX_RESPONSE_BYTES {
                    return Err(CliError::runtime("demo_response", "Demo response exceeds 1 MiB"));
                }
                bytes.extend_from_slice(&chunk);
            }
            let body: Value = serde_json::from_slice(&bytes)
                .map_err(|_| CliError::runtime("demo_response", "Server did not return valid JSON"))?;
            if !body["model"].is_string() || !body["answers"].is_object() || !body["usage"].is_object() {
                return Err(CliError::runtime("demo_response", "Server did not return a Jev systemone response (model, answers, usage)"));
            }
            let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
            if !global.quiet {
                writeln!(stderr, "Completed in {elapsed_ms:.1} ms").map_err(io_error)?;
            }
            let row = json!({"example": example.name, "elapsed_ms": elapsed_ms, "response": body, "metadata": metadata});
            if global.pretty {
                results.push(row);
            } else {
                output::write_json(stdout, &row, false).map_err(io_error)?;
                stdout.flush().map_err(io_error)?;
            }
        }
        if global.pretty {
            output::write_json(stdout, &results, true).map_err(io_error)?;
        }
        Ok(0)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_eight_examples_pass_the_real_server_request_parser() {
        let examples = examples("jev-latest");
        assert_eq!(examples.len(), 8);
        for example in examples {
            crate::server::jev::parse_request(&serde_json::to_vec(&example.request).unwrap())
                .unwrap();
        }
    }

    #[test]
    fn accepts_origin_or_v1_but_rejects_ambiguous_or_credential_urls() {
        for base in [
            "http://127.0.0.1:8080",
            "http://localhost/",
            "https://example.com/v1",
            "http://[::1]:8080/v1/",
        ] {
            assert_eq!(endpoint(base).unwrap().path(), "/v1/systemone");
        }
        for base in [
            "file:///tmp/server",
            "not a url",
            "http://user:secret@localhost",
            "http://localhost?key=secret",
            "http://localhost/#fragment",
            "http://localhost/other",
        ] {
            assert!(endpoint(base).is_err());
        }
    }
}
