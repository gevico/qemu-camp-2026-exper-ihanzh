use i2c_rs::{I2CEvent, I2CSlave};

const AT24C02_SIZE: usize = 256;
const AT24C02_PAGE_SIZE: usize = 8;

pub struct MockEeprom {
    addr: u8,
    regs: [u8; AT24C02_SIZE],
    pointer: u8,
    page_base: u8,
    page_off: u8,
    first_byte: bool,
}

impl MockEeprom {
    pub fn new(addr: u8) -> Self {
        Self {
            addr,
            regs: [0xFF; AT24C02_SIZE],
            pointer: 0,
            page_base: 0,
            page_off: 0,
            first_byte: true,
        }
    }
}

impl I2CSlave for MockEeprom {
    fn address(&self) -> u8 {
        self.addr
    }

    fn event(&mut self, event: I2CEvent) -> i32 {
        if event == I2CEvent::StartSend {
            self.first_byte = true;
        }
        0
    }

    fn send(&mut self, data: u8) -> i32 {
        if self.first_byte {
            self.pointer = data;
            self.page_base = data & !((AT24C02_PAGE_SIZE as u8) - 1);
            self.page_off = data & ((AT24C02_PAGE_SIZE as u8) - 1);
            self.first_byte = false;
        } else {
            let write_addr = self.page_base | self.page_off;
            self.regs[write_addr as usize] = data;

            self.page_off = (self.page_off + 1) & ((AT24C02_PAGE_SIZE as u8) - 1);
            self.pointer = self.page_base | self.page_off;
        }
        0
    }

    fn recv(&mut self) -> u8 {
        let val = self.regs[self.pointer as usize];
        self.pointer = self.pointer.wrapping_add(1);
        val
    }
}
