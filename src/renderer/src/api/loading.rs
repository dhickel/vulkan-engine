use std::time::Instant;

use super::errors::AssetError;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct LoadTicket(u64);

impl LoadTicket {
    pub(crate) fn new(raw: u64) -> Self {
        Self(raw)
    }

    pub(crate) fn raw(self) -> u64 {
        self.0
    }
}

pub enum LoadStatus<T> {
    Pending { queued_at: Instant },
    Uploaded { value: T },
    Failed { error: AssetError },
    Cancelled,
}
