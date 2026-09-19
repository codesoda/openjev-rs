use std::{fs, path::Path};

use serde_json::Value;

#[test]
fn exported_schema_is_the_plan_appendix() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let plan = fs::read_to_string(root.join("docs/PLAN.md")).unwrap();
    let appendix = plan
        .split_once("## Appendix A.")
        .unwrap()
        .1
        .split_once("```json\n")
        .unwrap()
        .1
        .split_once("\n```")
        .unwrap()
        .0;
    let documented: Value = serde_json::from_str(appendix).unwrap();
    let exported: Value = serde_json::from_str(
        &fs::read_to_string(root.join("schemas/readout-v1.schema.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(exported, documented);
    assert_eq!(exported["$id"], "urn:openjev:readout:v1");
}
