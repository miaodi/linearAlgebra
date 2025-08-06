#pragma once

namespace vec_ops {
template <typename IDX, typename VAL>
void copy_vec(const IDX size, VAL const *src, VAL *dst) {
  for (IDX i = 0; i < size; ++i) {
    dst[i] = src[i];
  }
}

template <typename IDX, typename VAL>
VAL vec_l2_norm(const IDX size, const VAL *vec) {
  VAL norm = static_cast<VAL>(0);
  for (IDX i = 0; i < size; ++i) {
    norm += vec[i] * vec[i];
  }
  return std::sqrt(norm);
}

template <typename IDX, typename VAL>
void scale_vec(const IDX size, const VAL alpha, VAL *vec) {
  for (IDX i = 0; i < size; ++i) {
    vec[i] *= alpha;
  }
}

template <typename IDX, typename VAL>
VAL dot_product(const IDX size, const VAL *vec1, const VAL *vec2) {
  VAL result = static_cast<VAL>(0);
  for (IDX i = 0; i < size; ++i) {
    result += vec1[i] * vec2[i];
  }
  return result;
}

// y = y + alpha * x
template <typename IDX, typename VAL>
void axpy(const IDX size, const VAL alpha, const VAL *x, VAL *y) {
  for (IDX i = 0; i < size; ++i) {
    y[i] += alpha * x[i];
  }
}
} // namespace vec_ops