// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "fpsan/fpsan.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <cctype>
#include <cstdint>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace {
using fpsan::Conversions;
using fpsan::Semantics;

struct Entry {
  std::string dtype;
  Semantics semantics;
  Conversions conversions;
  nb::object cls;
};

std::vector<Entry> &registry() {
  static std::vector<Entry> entries;
  return entries;
}

std::string lowered(std::string s) {
  for (char &c : s) {
    if (c == '-' || c == '.' || c == ' ')
      c = '_';
    else
      c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return s;
}

std::string dtype_name(nb::handle h) {
  if (nb::isinstance<nb::str>(h))
    return lowered(nb::cast<std::string>(h));
  if (nb::hasattr(h, "name"))
    return lowered(nb::cast<std::string>(h.attr("name")));
  if (nb::hasattr(h, "__name__"))
    return lowered(nb::cast<std::string>(h.attr("__name__")));
  return lowered(nb::cast<std::string>(nb::str(h)));
}

std::string normalize_dtype(nb::handle h) {
  std::string n = dtype_name(h);
  if (n == "float" || n == "float64" || n == "double" || n == "np_float64" || n == "numpy_float64")
    return "float64";
  if (n == "float32" || n == "single" || n == "f32" || n == "np_float32" || n == "numpy_float32")
    return "float32";
  if (n == "float16" || n == "half" || n == "f16" || n == "_float16" || n == "np_float16" ||
      n == "numpy_float16")
    return "float16";
  if (n == "bfloat16" || n == "bf16" || n == "__bf16")
    return "bfloat16";
  if (n == "fp8_e4m3" || n == "e4m3" || n == "float8_e4m3" || n == "fp8e4m3")
    return "fp8_e4m3";
  if (n == "fp8_e5m2" || n == "e5m2" || n == "float8_e5m2" || n == "fp8e5m2")
    return "fp8_e5m2";
  throw std::invalid_argument("unsupported fpsan dtype: " + n);
}

std::string semantics_suffix(Semantics s) { return fpsan::semantics_name(s); }

std::string conversions_suffix(Conversions c) {
  return c == Conversions::Explicit ? "Explicit" : "Implicit";
}

bool same_template(Semantics a, Semantics b) { return a == b; }

template <class F> std::uint64_t storage_bits(F v) {
  return static_cast<std::uint64_t>(v.to_storage_bits());
}

template <class F> std::uint64_t payload(F v) {
  static_assert(F::is_fpsan);
  return static_cast<std::uint64_t>(v.fpsan_payload());
}

template <class F> double to_double(F v) {
  static_assert(!F::is_algebraic);
  if constexpr (std::is_same_v<typename F::float_type, fpsan::fp8_e4m3> ||
                std::is_same_v<typename F::float_type, fpsan::fp8_e5m2>)
    return static_cast<double>(static_cast<float>(v.to_float()));
  else
    return static_cast<double>(v.to_float());
}

template <class FT> FT from_double(double v) {
  if constexpr (std::is_same_v<FT, fpsan::fp8_e4m3> || std::is_same_v<FT, fpsan::fp8_e5m2>)
    return FT(static_cast<float>(v));
  else
    return static_cast<FT>(v);
}

template <class F> std::string repr(const std::string &py_name, F v) {
  std::ostringstream os;
  os << "fpsan." << py_name << "(";
  if constexpr (F::is_algebraic)
    os << "payload=" << payload(v);
  else if constexpr (F::is_fpsan)
    os << to_double(v) << ", payload=" << payload(v);
  else
    os << to_double(v);
  os << ")";
  return os.str();
}

template <class F> void bind_value_ops(nb::class_<F> &cls) {
  cls.def(
         "__add__", [](F a, F b) { return a + b; }, nb::is_operator())
      .def(
          "__sub__", [](F a, F b) { return a - b; }, nb::is_operator())
      .def(
          "__mul__", [](F a, F b) { return a * b; }, nb::is_operator())
      .def(
          "__truediv__", [](F a, F b) { return a / b; }, nb::is_operator())
      .def(
          "__pos__", [](F a) { return +a; }, nb::is_operator())
      .def(
          "__neg__", [](F a) { return -a; }, nb::is_operator())
      .def(
          "__eq__", [](F a, F b) { return a == b; }, nb::is_operator())
      .def(
          "__ne__", [](F a, F b) { return a != b; }, nb::is_operator())
      .def(
          "__lt__", [](F a, F b) { return a < b; }, nb::is_operator())
      .def(
          "__le__", [](F a, F b) { return a <= b; }, nb::is_operator())
      .def(
          "__gt__", [](F a, F b) { return a > b; }, nb::is_operator())
      .def("__ge__", [](F a, F b) { return a >= b; }, nb::is_operator());
}

template <class F> void bind_math(nb::module_ &m) {
  m.def("exp", [](F x) { return fpsan::exp(x); });
  m.def("exp2", [](F x) { return fpsan::exp2(x); });
  m.def("exp10", [](F x) { return fpsan::exp10(x); });
  m.def("log", [](F x) { return fpsan::log(x); });
  m.def("log2", [](F x) { return fpsan::log2(x); });
  m.def("log10", [](F x) { return fpsan::log10(x); });
  m.def("sin", [](F x) { return fpsan::sin(x); });
  m.def("cos", [](F x) { return fpsan::cos(x); });
  m.def("sqrt", [](F x) { return fpsan::sqrt(x); });
  m.def("precise_sqrt", [](F x) { return fpsan::precise_sqrt(x); });
  m.def("rsqrt", [](F x) { return fpsan::rsqrt(x); });
  m.def("cbrt", [](F x) { return fpsan::cbrt(x); });
  m.def("erf", [](F x) { return fpsan::erf(x); });
  m.def("floor", [](F x) { return fpsan::floor(x); });
  m.def("ceil", [](F x) { return fpsan::ceil(x); });
  m.def("rcp", [](F x) { return fpsan::rcp(x); });
  m.def("fract", [](F x) { return fpsan::fract(x); });
  m.def("tanh", [](F x) { return fpsan::tanh(x); });
  m.def("fma", [](F a, F b, F c) { return fpsan::fma(a, b, c); });
  m.def("fmod", [](F a, F b) { return fpsan::fmod(a, b); });
  m.def("fmin", [](F a, F b) { return fpsan::fmin(a, b); });
  m.def("fmax", [](F a, F b) { return fpsan::fmax(a, b); });
  m.def("min", [](F a, F b) { return fpsan::min(a, b); });
  m.def("max", [](F a, F b) { return fpsan::max(a, b); });
  m.def("fmed3", [](F a, F b, F c) { return fpsan::fmed3(a, b, c); });
}

template <class F, bool EnableOps = true>
nb::object bind_value(nb::module_ &m, const std::string &py_name, const char *dtype) {
  using FT = typename F::float_type;

  auto cls =
      nb::class_<F>(m, py_name.c_str())
          .def(nb::init<>())
          .def(
              "__init__", [](F *self, double v) { new (self) F(from_double<FT>(v)); },
              nb::arg("value"))
          .def_static("from_storage_bits",
                      [](std::uint64_t bits) {
                        return F::from_storage_bits(static_cast<typename F::bits_type>(bits));
                      })
          .def_prop_ro("storage_bits", [](F v) { return storage_bits(v); })
          .def_prop_ro("semantics", [](F) { return F::semantics; })
          .def_prop_ro("conversions", [](F) { return F::conversions; })
          .def_prop_ro("dtype", [dtype](F) { return std::string(dtype); })
          .def("__repr__", [py_name](F v) { return repr(py_name, v); });

  if constexpr (EnableOps)
    bind_value_ops(cls);

  if constexpr (F::is_fpsan) {
    cls.def_static("from_payload", [](std::uint64_t p) {
      return F::from_fpsan_payload(static_cast<typename F::bits_type>(p));
    });
    cls.def_prop_ro("payload", [](F v) { return payload(v); });
  }
  if constexpr (!F::is_algebraic) {
    cls.def("to_float", [](F v) { return to_double(v); });
    cls.def("__float__", [](F v) { return to_double(v); });
  }

  if constexpr (EnableOps)
    bind_math<F>(m);

  return nb::borrow<nb::object>(cls);
}

template <class FT, Semantics S, Conversions C>
void add_one(nb::module_ &m, const char *dtype, const char *dtype_prefix) {
  using F = fpsan::Value<FT, S, C>;
  const bool is_canonical = C == Conversions::Explicit;
  const std::string base = std::string(dtype_prefix) + semantics_suffix(S);
  const std::string py_name = is_canonical ? base : base + conversions_suffix(C);
  constexpr bool enable_ops = !(S == Semantics::Native && (std::is_same_v<FT, fpsan::fp8_e4m3> ||
                                                           std::is_same_v<FT, fpsan::fp8_e5m2>));
  nb::object cls = bind_value<F, enable_ops>(m, py_name, dtype);
  registry().push_back(Entry{dtype, S, C, cls});
  if (is_canonical) {
    const std::string explicit_name = base + "Explicit";
    m.attr(explicit_name.c_str()) = cls;
  }
}

template <class FT, Semantics S>
void add_semantics(nb::module_ &m, const char *dtype, const char *dtype_prefix) {
  add_one<FT, S, Conversions::Explicit>(m, dtype, dtype_prefix);
  add_one<FT, S, Conversions::Implicit>(m, dtype, dtype_prefix);
}

template <class FT> void add_dtype(nb::module_ &m, const char *dtype, const char *dtype_prefix) {
  add_semantics<FT, Semantics::Native>(m, dtype, dtype_prefix);
  add_semantics<FT, Semantics::Triton>(m, dtype, dtype_prefix);
  add_semantics<FT, Semantics::Field>(m, dtype, dtype_prefix);
  add_semantics<FT, Semantics::Field2>(m, dtype, dtype_prefix);
  add_semantics<FT, Semantics::FieldFast>(m, dtype, dtype_prefix);
  add_semantics<FT, Semantics::FieldFast2>(m, dtype, dtype_prefix);
  add_semantics<FT, Semantics::FieldWithMulCasts>(m, dtype, dtype_prefix);
  add_semantics<FT, Semantics::FieldWithMulCasts2>(m, dtype, dtype_prefix);
  add_semantics<FT, Semantics::SophieGermainRing>(m, dtype, dtype_prefix);
  add_semantics<FT, Semantics::SophieGermainRing2>(m, dtype, dtype_prefix);
  add_semantics<FT, Semantics::PythagoreanRing>(m, dtype, dtype_prefix);
  add_semantics<FT, Semantics::PythagoreanRing2>(m, dtype, dtype_prefix);
}

nb::object value_type(nb::handle dtype, Semantics semantics, Conversions conversions) {
  const std::string normalized = normalize_dtype(dtype);
  for (const Entry &entry : registry()) {
    if (entry.dtype == normalized && same_template(entry.semantics, semantics) &&
        entry.conversions == conversions)
      return entry.cls;
  }
  throw std::invalid_argument("unsupported fpsan Value template combination");
}

} // namespace

NB_MODULE(_fpsan, m) {
  m.doc() = "Python bindings for hip-fpsan scalar Value types";

  nb::enum_<Semantics>(m, "Semantics")
      .value("Native", Semantics::Native)
      .value("Triton", Semantics::Triton)
      .value("Field", Semantics::Field)
      .value("Field2", Semantics::Field2)
      .value("FieldFast", Semantics::FieldFast)
      .value("FieldFast2", Semantics::FieldFast2)
      .value("FieldWithMulCasts", Semantics::FieldWithMulCasts)
      .value("FieldWithMulCasts2", Semantics::FieldWithMulCasts2)
      .value("SophieGermainRing", Semantics::SophieGermainRing)
      .value("SophieGermainRing2", Semantics::SophieGermainRing2)
      .value("PythagoreanRing", Semantics::PythagoreanRing)
      .value("PythagoreanRing2", Semantics::PythagoreanRing2);

  nb::enum_<Conversions>(m, "Conversions")
      .value("Explicit", Conversions::Explicit)
      .value("Implicit", Conversions::Implicit);

  add_dtype<float>(m, "float32", "Float32");
  add_dtype<double>(m, "float64", "Float64");
#if FPSAN_HAS_FLOAT16
  add_dtype<_Float16>(m, "float16", "Float16");
#endif
#if FPSAN_HAS_BF16
  add_dtype<__bf16>(m, "bfloat16", "BFloat16");
#endif
  add_dtype<fpsan::fp8_e4m3>(m, "fp8_e4m3", "FP8E4M3");
  add_dtype<fpsan::fp8_e5m2>(m, "fp8_e5m2", "FP8E5M2");

  m.def(
      "value_type",
      [](nb::handle dtype, Semantics semantics, Conversions conversions) {
        return value_type(dtype, semantics, conversions);
      },
      nb::arg("dtype"), nb::arg("semantics"), nb::arg("conversions") = Conversions::Explicit,
      "Return the bound Value class for a dtype, Semantics, and Conversions.");

  m.def(
      "available_value_types",
      []() {
        nb::list out;
        for (const Entry &entry : registry()) {
          nb::dict d;
          d["dtype"] = entry.dtype;
          d["semantics"] = entry.semantics;
          d["conversions"] = entry.conversions;
          d["type"] = entry.cls;
          out.append(d);
        }
        return out;
      },
      "Return dictionaries describing the bound Value template instantiations.");
}
