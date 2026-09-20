import assert from "node:assert/strict";
import { TypeSafeClient, choice, noul, score } from "@typesafe-ai/sdk";

const baseURL = process.env.OPENJEV_BASE_URL ?? "http://127.0.0.1:8080";
const apiKey = process.env.OPENJEV_API_KEY ?? "openjev-local-sdk-smoke";
const client = new TypeSafeClient({
  apiKey,
  baseURL,
  timeout: 120_000,
  retry: { maxRetries: 0 },
});

const models = await client.models.list();
if (models.length !== 1) throw new Error(`expected one resident model, got ${models.length}`);

const result = await client.systemOne({
  model: "jev-latest",
  state: { ticket: "A duplicate charge needs review", severity: 3 },
  questions: {
    route: choice("Which queue should own this?", {
      billing: "Payment and duplicate-charge problems",
      support: "General product support",
    }),
    review: noul("Does this need human review?", {
      true: "A human should inspect it",
      false: "Automation can safely close it",
    }),
    urgency: score("How urgent is this?", ["low", "medium", "high"]),
  },
});

assert.equal(typeof models[0].description, "string");
assert.equal(typeof models[0].release_date, "string");
assert.equal(result.model, models[0].name);
assert.deepEqual(Object.keys(result.answers), ["route", "review", "urgency"]);
const probability = (value) => {
  assert.equal(typeof value, "number");
  assert.ok(Number.isFinite(value) && value >= 0 && value <= 1);
  assert.ok(Math.abs(value * 100 - Math.round(value * 100)) < 1e-9);
};
const distribution = (values, labels) => {
  assert.deepEqual(Object.keys(values), labels);
  Object.values(values).forEach(probability);
  // Independent two-decimal rounding need not sum to precisely one.
  assert.ok(Math.abs(Object.values(values).reduce((a, b) => a + b, 0) - 1) <= labels.length * 0.005 + 1e-9);
};
const { route, review, urgency } = result.answers;
assert.equal(route.type, "choice");
assert.ok(["billing", "support"].includes(route.choice));
probability(route.confidence);
distribution(route.probabilities, ["billing", "support"]);
assert.equal(review.type, "noul");
probability(review.noul);
assert.equal(urgency.type, "score");
assert.ok(Number.isFinite(urgency.score) && urgency.score >= 0 && urgency.score <= 2);
assert.ok(Math.abs(urgency.score * 100 - Math.round(urgency.score * 100)) < 1e-9);
assert.deepEqual(urgency.legend, { "0": "low", "1": "medium", "2": "high" });
probability(urgency.confidence);
distribution(urgency.probabilities, ["0", "1", "2"]);
assert.ok(Number.isSafeInteger(result.usage.input_tokens) && result.usage.input_tokens > 0);
assert.equal(result.usage.output_tokens, 0);

let rejected = false;
try {
  await client.systemOne({
    model: "not-the-loaded-model",
    state: "x",
    questions: { q: noul("?") },
  });
} catch (error) {
  rejected = error?.status === 404;
}
if (!rejected) throw new Error("unknown model did not produce the SDK's 404 error path");

let floatRejected = false;
try {
  await client.systemOne({
    state: { unsupportedFloat: 1.5 },
    questions: { q: noul("?") },
  });
} catch (error) {
  floatRejected = error?.status === 422;
}
assert.ok(floatRejected, "unsupported floats must produce the SDK's 422 error path");

console.log(JSON.stringify({
  sdk: "@typesafe-ai/sdk@0.6.0",
  source_commit: "66880ccded6cb642dc1809620c2b108c33730214",
  baseURL,
  model: result.model,
  answers: Object.keys(result.answers),
  usage: result.usage,
  status: "passed",
}));
