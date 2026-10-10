import 'dart:typed_data';

final _scratch = Float32List(1);

/// Round to float32, the Dart equivalent of JS Math.fround: applied after
/// every arithmetic step to reproduce numpy's float32 results exactly.
double f(double x) {
  _scratch[0] = x;
  return _scratch[0];
}
