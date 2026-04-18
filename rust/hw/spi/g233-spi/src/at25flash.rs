const AT25_CMD_WREN: u8 = 0x06;     // Write Enable
const AT25_CMD_RDSR: u8 = 0x05;     // Read Status Register
const AT25_CMD_READ: u8 = 0x03;     // Read Data
const AT25_CMD_WRITE: u8 = 0x02;    // Write Data

const AT25_SR_WEL: u8 = 1u8 << 1;   // Write Enable Latch

#[derive(Debug, Copy, Clone)]
pub enum At25State {
    Idle,
    ReadStatus,
    ReadAddr,
    ReadData { addr: u8 },
    WriteAddr,
    WriteData { addr: u8 },
}

#[derive(Debug)]
pub struct At25flash {
    mem: [u8; 256],
    wel: bool,
    state: At25State,
}

impl Default for At25flash {
    fn default() -> Self {
        Self {
            mem: [0xff; 256],
            wel: false,
            state: At25State::Idle,
        }
    }
}

impl At25flash {
    fn status(&self) -> u8 {
        if self.wel { AT25_SR_WEL } else { 0 }
    }

    pub fn xfer(&mut self, tx: u8) -> u8 {
        match tx {
            AT25_CMD_WREN => {
                self.wel = true;
                self.state = At25State::Idle;
                return 0;
            }
            AT25_CMD_RDSR => {
                self.state = At25State::ReadStatus;
                return 0;
            }
            AT25_CMD_READ => {
                self.state = At25State::ReadAddr;
                return 0;
            }
            AT25_CMD_WRITE => {
                self.state = At25State::WriteAddr;
                return 0;
            }
            _ => {}
        }

        match self.state {
            At25State::Idle => 0,
            At25State::ReadStatus => self.status(),
            At25State::ReadAddr => {
                self.state = At25State::ReadData { addr: tx };
                0
            }
            At25State::ReadData { addr } => {
                let out = self.mem[addr as usize];
                self.state = At25State::ReadData {
                    addr: addr.wrapping_add(1),
                };
                out
            }
            At25State::WriteAddr => {
                self.state = At25State::WriteData { addr: tx };
                0
            }
            At25State::WriteData { addr } => {
                if self.wel {
                    self.mem[addr as usize] = tx;
                    self.state = At25State::WriteData {
                        addr: addr.wrapping_add(1),
                    };
                }
                0
            }
        }
    }
}
