use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

pub const MAX_BINARY_MESSAGE_BYTES: usize = 16 * 1024 * 1024;
const MAX_HEADER_BYTES: usize = 64 * 1024;
const MAX_BUFFERS: usize = 1024;

#[derive(Debug, Default)]
pub struct BinaryMessage {
    pub message: Value,
    pub buffers: Vec<Vec<u8>>,
}

#[derive(Serialize, Deserialize)]
struct Header {
    message: Value,
    lengths: Vec<usize>,
}

impl BinaryMessage {
    pub fn new(message: Value) -> Self {
        Self {
            message,
            buffers: Vec::new(),
        }
    }

    pub fn attach(&mut self, bytes: Vec<u8>) -> Result<Value, String> {
        if self.buffers.len() >= MAX_BUFFERS
            || self
                .buffers
                .iter()
                .map(Vec::len)
                .sum::<usize>()
                .saturating_add(bytes.len())
                > MAX_BINARY_MESSAGE_BYTES - MAX_HEADER_BYTES - 4
        {
            return Err("binary message capacity exceeded".into());
        }
        let index = self.buffers.len();
        self.buffers.push(bytes);
        Ok(json!({"$binary": index}))
    }

    pub fn buffer(&self, value: &Value) -> Result<&[u8], String> {
        let index = value
            .as_object()
            .filter(|object| object.len() == 1)
            .and_then(|object| object.get("$binary"))
            .and_then(Value::as_u64)
            .and_then(|index| usize::try_from(index).ok())
            .ok_or("invalid binary reference")?;
        self.buffers
            .get(index)
            .map(Vec::as_slice)
            .ok_or_else(|| "missing binary buffer".into())
    }

    pub fn encode(self) -> Result<Vec<u8>, String> {
        let header = serde_json::to_vec(&Header {
            message: self.message,
            lengths: self.buffers.iter().map(Vec::len).collect(),
        })
        .map_err(|error| error.to_string())?;
        let size = 4 + header.len() + self.buffers.iter().map(Vec::len).sum::<usize>();
        if header.len() > MAX_HEADER_BYTES
            || self.buffers.len() > MAX_BUFFERS
            || size > MAX_BINARY_MESSAGE_BYTES
        {
            return Err("binary message capacity exceeded".into());
        }
        let mut output = Vec::with_capacity(size);
        output.extend_from_slice(&(header.len() as u32).to_be_bytes());
        output.extend_from_slice(&header);
        for buffer in self.buffers {
            output.extend_from_slice(&buffer);
        }
        Ok(output)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 4 || bytes.len() > MAX_BINARY_MESSAGE_BYTES {
            return Err("invalid binary message size".into());
        }
        let length = u32::from_be_bytes(bytes[..4].try_into().expect("four bytes")) as usize;
        if length > MAX_HEADER_BYTES || length > bytes.len() - 4 {
            return Err("invalid binary header size".into());
        }
        let header: Header =
            serde_json::from_slice(&bytes[4..4 + length]).map_err(|error| error.to_string())?;
        if header.lengths.len() > MAX_BUFFERS {
            return Err("too many binary buffers".into());
        }
        let mut offset = 4 + length;
        let mut buffers = Vec::with_capacity(header.lengths.len());
        for length in header.lengths {
            if length > bytes.len() - offset {
                return Err("truncated binary buffer".into());
            }
            buffers.push(bytes[offset..offset + length].to_vec());
            offset += length;
        }
        if offset != bytes.len() {
            return Err("trailing binary data".into());
        }
        Ok(Self {
            message: header.message,
            buffers,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_payloads_roundtrip_without_text_encoding() {
        let mut input = BinaryMessage::default();
        let reference = input.attach(vec![0, 255, 128]).unwrap();
        input.message = json!({"data": reference, "meta": {"codec":"arbitrary"}});
        let output = BinaryMessage::decode(&input.encode().unwrap()).unwrap();
        assert_eq!(
            output.buffer(&output.message["data"]).unwrap(),
            &[0, 255, 128]
        );
        assert_eq!(output.message["meta"]["codec"], "arbitrary");
        assert!(output.buffer(&json!({"$binary":1})).is_err());
        assert!(output.buffer(&json!({"$binary":0,"extra":true})).is_err());
    }

    #[test]
    fn malformed_lengths_and_trailing_bytes_are_rejected() {
        for bytes in [vec![], vec![0, 0, 0, 255], vec![255, 255, 255, 255]] {
            assert!(BinaryMessage::decode(&bytes).is_err());
        }
        let mut input = BinaryMessage::default();
        input.attach(vec![1, 2, 3]).unwrap();
        let bytes = input.encode().unwrap();
        for end in 0..bytes.len() {
            assert!(BinaryMessage::decode(&bytes[..end]).is_err());
        }
        let mut trailing = bytes;
        trailing.push(0);
        assert!(BinaryMessage::decode(&trailing).is_err());
    }
}
