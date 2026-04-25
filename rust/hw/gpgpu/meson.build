# GPGPU 设备的 Meson 构建配置
#
# 这个文件告诉 Meson 构建系统如何编译 GPGPU 设备
# system_ss 是系统模拟器的源文件集合

_gpgpu_core_c = static_library(
  'gpgpu_core_c',
  files('../../../hw/gpgpu/gpgpu_core.c'),
  c_args: ['-DGPGPU_CORE_USE_ACCESSORS'],
)

_libgpgpu_rs = static_library(
  'gpgpu_rs',
  files('src/lib.rs', 'src/device.rs'),
  override_options: ['rust_std=2021', 'build.rust_std=2021'],
  rust_abi: 'rust',
  link_with: [
    _gpgpu_core_c,
    _util_rs,
    _migration_rs,
    _bql_rs,
    _qom_rs,
    _system_rs,
    _hwcore_rs,
    _trace_rs,
  ],
  dependencies: [common_rs],
)

rust_devices_ss.add(when: 'CONFIG_X_GPGPU_RUST', if_true: [declare_dependency(
  link_whole: [_libgpgpu_rs],
  variables: {'crate': 'gpgpu_rs'},
)])
