use std::collections::HashSet;

use crate::types::{OpenJevError, Result};

pub const LETTERS: &str = "ABCDEFGHIJKLMNOP";

/// Minimal tokenizer surface needed to validate answer slots.
pub trait SlotTokenizer {
    fn tokenize_no_bos(&self, text: &str) -> Result<Vec<u32>>;
    fn token_piece(&self, token_id: u32) -> Result<Vec<u8>>;
    fn vocabulary_size(&self) -> usize;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedSlots {
    pub prompt_token_ids: Vec<u32>,
    pub answer_token_ids: Vec<u32>,
}

pub fn verify_slots<T: SlotTokenizer>(
    tokenizer: &T,
    prompt: &str,
    option_count: usize,
) -> Result<VerifiedSlots> {
    if !(2..=16).contains(&option_count) {
        return Err(OpenJevError::Slot(
            "answer slots require 2-16 options".to_owned(),
        ));
    }
    let prompt_token_ids = tokenizer.tokenize_no_bos(prompt)?;
    if prompt_token_ids.is_empty() {
        return Err(OpenJevError::Slot(
            "tokenized prompt must not be empty".to_owned(),
        ));
    }
    let mut answer_token_ids = Vec::with_capacity(option_count);
    let mut unique = HashSet::with_capacity(option_count);
    for letter in LETTERS.bytes().take(option_count) {
        let text = char::from(letter).to_string();
        let encoded = tokenizer.tokenize_no_bos(&text)?;
        if encoded.len() != 1 {
            return Err(OpenJevError::Slot(format!(
                "answer slot {text:?} is not exactly one token"
            )));
        }
        let token_id = encoded[0];
        if usize::try_from(token_id).map_or(true, |id| id >= tokenizer.vocabulary_size()) {
            return Err(OpenJevError::Slot(format!(
                "answer slot {text:?} token ID {token_id} is outside the vocabulary"
            )));
        }
        if !unique.insert(token_id) {
            return Err(OpenJevError::Slot(format!(
                "answer slot {text:?} collides with an earlier token"
            )));
        }
        if tokenizer.token_piece(token_id)? != [letter] {
            return Err(OpenJevError::Slot(format!(
                "answer slot {text:?} does not round-trip to exact ASCII"
            )));
        }
        let combined = tokenizer.tokenize_no_bos(&format!("{prompt}{text}"))?;
        let mut expected = prompt_token_ids.clone();
        expected.push(token_id);
        if combined != expected {
            return Err(OpenJevError::Slot(format!(
                "answer boundary changes tokenization for slot {text}"
            )));
        }
        answer_token_ids.push(token_id);
    }
    Ok(VerifiedSlots {
        prompt_token_ids,
        answer_token_ids,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    struct FakeTokenizer {
        pieces: HashMap<String, Vec<u32>>,
        reverse: HashMap<u32, Vec<u8>>,
        vocab: usize,
    }

    impl SlotTokenizer for FakeTokenizer {
        fn tokenize_no_bos(&self, text: &str) -> Result<Vec<u32>> {
            self.pieces.get(text).cloned().ok_or_else(|| {
                OpenJevError::Slot(format!("missing fake tokenization for {text:?}"))
            })
        }

        fn token_piece(&self, token_id: u32) -> Result<Vec<u8>> {
            self.reverse
                .get(&token_id)
                .cloned()
                .ok_or_else(|| OpenJevError::Slot("missing fake piece".to_owned()))
        }

        fn vocabulary_size(&self) -> usize {
            self.vocab
        }
    }

    fn tokenizer() -> FakeTokenizer {
        FakeTokenizer {
            pieces: HashMap::from([
                ("prompt".to_owned(), vec![7]),
                ("A".to_owned(), vec![10]),
                ("B".to_owned(), vec![11]),
                ("promptA".to_owned(), vec![7, 10]),
                ("promptB".to_owned(), vec![7, 11]),
            ]),
            reverse: HashMap::from([(10, b"A".to_vec()), (11, b"B".to_vec())]),
            vocab: 20,
        }
    }

    #[test]
    fn verifies_exact_slots() {
        let slots = verify_slots(&tokenizer(), "prompt", 2).unwrap();
        assert_eq!(slots.answer_token_ids, [10, 11]);
    }

    #[test]
    fn rejects_roundtrip_collision_and_boundary_changes() {
        let mut bad = tokenizer();
        bad.reverse.insert(10, b" a".to_vec());
        assert!(
            verify_slots(&bad, "prompt", 2)
                .unwrap_err()
                .to_string()
                .contains("round-trip")
        );

        let mut bad = tokenizer();
        bad.pieces.insert("B".to_owned(), vec![10]);
        bad.pieces.insert("promptB".to_owned(), vec![7, 10]);
        assert!(
            verify_slots(&bad, "prompt", 2)
                .unwrap_err()
                .to_string()
                .contains("collides")
        );

        let mut bad = tokenizer();
        bad.pieces.insert("promptB".to_owned(), vec![99]);
        assert!(
            verify_slots(&bad, "prompt", 2)
                .unwrap_err()
                .to_string()
                .contains("boundary")
        );
    }

    #[test]
    fn rejects_multitoken_and_vocabulary_boundary() {
        let mut bad = tokenizer();
        bad.pieces.insert("A".to_owned(), vec![10, 12]);
        assert!(
            verify_slots(&bad, "prompt", 2)
                .unwrap_err()
                .to_string()
                .contains("one token")
        );

        let mut bad = tokenizer();
        bad.vocab = 11;
        assert!(
            verify_slots(&bad, "prompt", 2)
                .unwrap_err()
                .to_string()
                .contains("vocabulary")
        );
    }
}
