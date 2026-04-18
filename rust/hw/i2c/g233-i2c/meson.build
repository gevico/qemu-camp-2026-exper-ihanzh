_libg233_i2c_rs = static_library(
  'g233_i2c',
  files('src/lib.rs', 'src/device.rs', 'src/at24c02.rs'),
  override_options: ['rust_std=2021', 'build.rust_std=2021'],
  rust_abi: 'rust',
  link_with: [
    _util_rs,
    _migration_rs,
    _bql_rs,
    _qom_rs,
    _system_rs,
    _hwcore_rs,
    _trace_rs,
  ],
  dependencies: [common_rs, i2c_rs],
)

rust_devices_ss.add(when: 'CONFIG_X_I2C_RUST', if_true: [declare_dependency(
  link_whole: [_libg233_i2c_rs],
  variables: {'crate': 'g233_i2c'},
)])
