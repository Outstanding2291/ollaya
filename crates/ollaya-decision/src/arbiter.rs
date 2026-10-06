//! `arbiter-fixed-v1`: Arbiter by Codekins (a LoRA on Gemma 3 4B IT plus a fixed 24-slot head),
//! whose rows are its training prompt (`ollaya_convert.families.arbiter.layout`).
//!
//! Every question is one causal row, the prompt after Gemma's `<bos>`:
//!
//! ```text
//! [bos] tok("State: {render(state)}\n\nQuestion: {render(instructions)}\n\n"
//!           "Options:\n{block}\n\nAnswer:")
//!   noul    block = "T. Yes / True\nF. No / False"
//!   choice  block = "A. {option 0}\nB. {option 1}\n..."   1..=16 options, letters A..P
//!   score   block = "0\n1\n2\n3\n4\n5"                    exactly 6 levels
//! ```
//!
//! A choice option is `name` or `name: render(description)`; the noul and score descriptions are
//! not part of the prompt. The graph returns the head's 24 raw slot scores at the row's last
//! position; a question's option logits are the scores at its slots (`decision.json` `slots`):
//! noul `[F, T]` (false, true), choice `A..`, score `0..5`. Requests the fixed head cannot answer
//! (a choice over 16 options, a score with other than 6 levels, a row over `max_row_tokens`) are
//! rejected, never truncated.

use std::fmt::Write;

use serde::Deserialize;
use serde_json::Value;

use crate::Error;
use crate::kev::{option_text, render};
use crate::layout::TokenEncoder;
use crate::question::{QType, Question, Questions};

pub const LAYOUT: &str = "arbiter-fixed-v1";

/// The choice letters, in slot order.
const LETTERS: &[u8; 16] = b"ABCDEFGHIJKLMNOP";
const NOUL_BLOCK: &str = "T. Yes / True\nF. No / False";
const SCORE_BLOCK: &str = "0\n1\n2\n3\n4\n5";
const SCORE_LEVELS: usize = 6;

/// Token ids, as `decision.json` declares them.
#[derive(Debug, Clone, Deserialize)]
pub struct ArbiterTokens {
    /// Starts every row (`<bos>`).
    pub bos: u32,
    pub pad: u32,
}

/// The head slots of each question type's option logits, in Ollaya's option order.
#[derive(Debug, Clone, Deserialize)]
pub struct ArbiterSlots {
    /// `[false, true]`: the slots of `F` and `T`.
    pub noul: Vec<usize>,
    /// Option j of a choice reads `choice[j]` (letter j).
    pub choice: Vec<usize>,
    /// Level j of a score reads `score[j]` (digit j).
    pub score: Vec<usize>,
}

/// The layout as `decision.json` declares it.
#[derive(Debug, Clone, Deserialize)]
pub struct ArbiterLayout {
    /// A longer row is rejected.
    pub max_row_tokens: usize,
    pub special_tokens: ArbiterTokens,
    /// Scores per row in the graph's output.
    pub num_slots: usize,
    pub slots: ArbiterSlots,
}

/// One question's row.
#[derive(Debug, Clone, PartialEq)]
pub struct ArbiterRow {
    pub ids: Vec<u32>,
    /// Where the head is read (the last position).
    pub last_pos: usize,
    /// The slots of the question's option logits, in option order.
    pub slots: Vec<usize>,
}

impl ArbiterLayout {
    /// Reject configurations this layout cannot run.
    pub fn validate(&self) -> Result<(), Error> {
        let s = &self.slots;
        let ok = self.max_row_tokens > 0
            && s.noul.len() == 2
            && (1..=LETTERS.len()).contains(&s.choice.len())
            && s.score.len() == SCORE_LEVELS
            && s.noul
                .iter()
                .chain(&s.choice)
                .chain(&s.score)
                .all(|&i| i < self.num_slots);
        if ok {
            Ok(())
        } else {
            Err(Error::invalid(format!(
                "max_row_tokens={} and slots {s:?} do not fit a {}-slot head",
                self.max_row_tokens, self.num_slots
            )))
        }
    }

    /// Most options a choice can have.
    pub fn max_options(&self) -> usize {
        self.slots.choice.len()
    }

    /// One question's instructions, option block and slots, validated as the reference does.
    pub fn render_question(
        &self,
        qid: &str,
        q: &Question,
    ) -> Result<(String, String, Vec<usize>), Error> {
        let bad = |msg: &str| Error::invalid(format!("question {qid:?}: {msg}"));
        let criteria = q.definition.get("criteria").filter(|c| !c.is_null());
        let (block, slots) = match q.qtype {
            QType::Noul => {
                if criteria.is_some_and(|c| !c.is_object()) {
                    return Err(bad("noul criteria must be an object"));
                }
                (NOUL_BLOCK.to_owned(), self.slots.noul.clone())
            }
            QType::Choice => {
                let m = match criteria {
                    Some(Value::Object(m)) if !m.is_empty() => m,
                    _ => return Err(bad("choice criteria must be a non-empty object")),
                };
                if m.len() > self.max_options() {
                    return Err(Error::TooManyOptions {
                        question: qid.to_owned(),
                        options: m.len(),
                        head_max_len: self.max_options(),
                    });
                }
                let mut block = String::new();
                for (i, (name, d)) in m.iter().enumerate() {
                    let sep = if i > 0 { "\n" } else { "" };
                    let letter = char::from(LETTERS[i]);
                    let _ = write!(block, "{sep}{letter}. {}", option_text(name, Some(d)));
                }
                (block, self.slots.choice[..m.len()].to_vec())
            }
            QType::Score => match criteria.and_then(Value::as_array).map(Vec::len) {
                Some(SCORE_LEVELS) => (SCORE_BLOCK.to_owned(), self.slots.score.clone()),
                Some(n) => return Err(bad(&format!("exactly 6 levels expected, got {n}"))),
                None => return Err(bad("score criteria must be a list of 6 levels")),
            },
        };
        let instructions = render(q.definition.get("instructions").unwrap_or(&Value::Null));
        Ok((instructions, block, slots))
    }

    /// Every question's row, in request order. All questions are validated before any is tokenized.
    pub fn rows(
        &self,
        enc: &dyn TokenEncoder,
        state: &Value,
        questions: &Questions,
    ) -> Result<Vec<ArbiterRow>, Error> {
        let rendered = questions
            .iter()
            .map(|(qid, q)| self.render_question(qid, q))
            .collect::<Result<Vec<_>, _>>()?;
        let state_text = render(state);
        questions
            .keys()
            .zip(rendered)
            .map(|(qid, (instructions, block, slots))| {
                let prompt = format!("State: {state_text}\n\nQuestion: {instructions}\n\n");
                let text = format!("{prompt}Options:\n{block}\n\nAnswer:");
                let mut ids = vec![self.special_tokens.bos];
                ids.extend(enc.encode(&text)?);
                if ids.len() > self.max_row_tokens {
                    return Err(Error::invalid(format!(
                        "question {qid:?}: the row is {} tokens; this model reads up to {}",
                        ids.len(),
                        self.max_row_tokens
                    )));
                }
                Ok(ArbiterRow {
                    last_pos: ids.len() - 1,
                    ids,
                    slots,
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// One id per char, so rows can be read back.
    struct Chars;

    impl TokenEncoder for Chars {
        fn encode(&self, text: &str) -> Result<Vec<u32>, Error> {
            Ok(text.chars().map(u32::from).collect())
        }
    }

    fn text(ids: &[u32]) -> String {
        ids.iter()
            .map(|&i| char::from_u32(i).unwrap_or('#'))
            .collect()
    }

    fn layout(max_row_tokens: usize) -> ArbiterLayout {
        serde_json::from_value(json!({
            "max_row_tokens": max_row_tokens,
            "special_tokens": {"bos": 2, "pad": 0},
            "num_slots": 24,
            "slots": {"noul": [1, 0],
                      "choice": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
                      "score": [18, 19, 20, 21, 22, 23]},
        }))
        .unwrap()
    }

    fn rows(l: &ArbiterLayout, state: Value, questions: Value) -> Result<Vec<ArbiterRow>, Error> {
        l.rows(&Chars, &state, &crate::parse_questions(&questions).unwrap())
    }

    /// A choice with `n` options named `o0`, `o1`, ...
    fn choice(n: usize) -> Value {
        let mut criteria = serde_json::Map::new();
        for i in 0..n {
            criteria.insert(format!("o{i}"), Value::Null);
        }
        json!({"type": "choice", "instructions": "x", "criteria": criteria})
    }

    #[test]
    fn builds_the_training_prompt() {
        let l = layout(8192);
        let state = json!({"order": "A-104", "amount": 1.0});
        let questions = json!({
            "n": {"type": "noul", "instructions": "Refund?", "criteria": {"true": "money back"}},
            "c": {"type": "choice", "instructions": {"task": "route"},
                  "criteria": {"billing": "charges", "tech": {"covers": ["bugs"]},
                               "other": null, "x": ""}},
            "s": {"type": "score", "instructions": "How upset?",
                  "criteria": ["calm", "mild", "annoyed", "upset", "angry", "furious"]},
        });
        let r = rows(&l, state, questions).unwrap();
        let head = "State: order: A-104\namount: 1.0\n\nQuestion: ";
        assert_eq!(r[0].ids[0], 2);
        let noul = "Refund?\n\nOptions:\nT. Yes / True\nF. No / False\n\nAnswer:";
        assert_eq!(text(&r[0].ids[1..]), format!("{head}{noul}"));
        let choice = "A. billing: charges\nB. tech: covers:\n  - bugs\nC. other\nD. x";
        assert_eq!(
            text(&r[1].ids[1..]),
            format!("{head}task: route\n\nOptions:\n{choice}\n\nAnswer:")
        );
        let score = "How upset?\n\nOptions:\n0\n1\n2\n3\n4\n5\n\nAnswer:";
        assert_eq!(text(&r[2].ids[1..]), format!("{head}{score}"));
        assert!(r.iter().all(|row| row.last_pos == row.ids.len() - 1));
        assert_eq!(r[0].slots, [1, 0]);
        assert_eq!(r[1].slots, [2, 3, 4, 5]);
        assert_eq!(r[2].slots, [18, 19, 20, 21, 22, 23]);
    }

    #[test]
    fn takes_up_to_sixteen_options() {
        let q = json!({"q": choice(16)});
        let r = rows(&layout(8192), json!("s"), q).unwrap();
        assert_eq!(r[0].slots, (2..18).collect::<Vec<_>>());
        assert!(text(&r[0].ids).ends_with("\nP. o15\n\nAnswer:"));
    }

    #[test]
    fn rejects_what_the_head_cannot_answer() {
        let l = layout(8192);
        let ok = json!({"type": "noul", "instructions": "fine?"});
        let many = rows(&l, json!("s"), json!({"ok": ok, "q": choice(17)}));
        assert!(matches!(
            many,
            Err(Error::TooManyOptions {
                options: 17,
                head_max_len: 16,
                ..
            })
        ));
        for n in [5, 7] {
            let levels: Vec<String> = (0..n).map(|i| format!("level {i}")).collect();
            let q = json!({"type": "score", "instructions": "x", "criteria": levels});
            let r = rows(&l, json!("s"), json!({"ok": ok, "q": q}));
            assert!(matches!(r, Err(Error::Invalid(m)) if m.contains("exactly 6 levels")));
        }
        let list = json!({"type": "choice", "instructions": "x", "criteria": ["a", "b"]});
        let r = rows(&l, json!("s"), json!({"q": list}));
        assert!(matches!(r, Err(Error::Invalid(m)) if m.contains("non-empty object")));
        let r = rows(&layout(40), json!("y".repeat(40)), json!({"q": ok}));
        assert!(matches!(r, Err(Error::Invalid(m)) if m.contains("reads up to 40")));
    }

    #[test]
    fn validates_the_slots() {
        layout(8192).validate().unwrap();
        let mut l = layout(8192);
        l.slots.score.push(24);
        assert!(l.validate().is_err());
        let mut l = layout(8192);
        l.slots.choice = (2..19).collect();
        assert!(l.validate().is_err());
    }
}
