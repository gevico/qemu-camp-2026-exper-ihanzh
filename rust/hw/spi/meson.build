_libspi_rs = static_library(
  'spi_rs',
  files('src/lib.rs'),
  override_options: ['rust_std=2021', 'build.rust_std=2021'],
  rust_abi: 'rust',
  dependencies: [common_rs],
)

spi_rs = declare_dependency(
  link_whole: [_libspi_rs],
  variables: {'crate': 'spi_rs'},
)

rust_devices_ss.add(when: 'CONFIG_X_SPI_RUST', if_true: [spi_rs])

subdir('g233-spi')
