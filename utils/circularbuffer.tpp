/*
 CircularBuffer.tpp - Circular buffer library for Arduino.
 Copyright (c) 2017 Roberto Lo Giacco.

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU Lesser General Public License as
 published by the Free Software Foundation, either version 3 of the
 License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

namespace utils {
template <typename T>
constexpr CircularBuffer<T>::CircularBuffer(const size_t S)
    : _buffer(S), _head(_buffer.begin()), _tail(_buffer.begin()), _count(0)
{
}

template <typename T> void CircularBuffer<T>::push_front(const T &value) {
  if (!available()) {
    resize(_buffer.size() == 0 ? 1 : (_buffer.size() * 2));
  }
  push_front_overwrite(value);
}

template <typename T> bool CircularBuffer<T>::push_front_overwrite(const T &value) {
  if (_head == _buffer.begin()) {
    _head = _buffer.end();
  }
  *--_head = value;
  if (_count == _buffer.size()) {
    if (_tail == _buffer.begin()) {
      _tail = _buffer.end();
    }
    _tail--;
    return false;
  } else {
    if (_count++ == 0) {
      _tail = _head;
    }
    return true;
  }
}

template <typename T> void CircularBuffer<T>::push_front(T &&value) {
  if (!available()) {
    resize(_buffer.size() == 0 ? 1 : (_buffer.size() * 2));
  }
  push_front_overwrite(std::move(value));
}

template <typename T> bool CircularBuffer<T>::push_front_overwrite(T &&value) {
  if (_head == _buffer.begin()) {
    _head = _buffer.end();
  }
  *--_head = std::move(value);
  if (_count == _buffer.size()) {
    if (_tail == _buffer.begin()) {
      _tail = _buffer.end();
    }
    _tail--;
    return false;
  } else {
    if (_count++ == 0) {
      _tail = _head;
    }
    return true;
  }
}

template <typename T> void CircularBuffer<T>::push_back(const T &value) {
  if (!available()) {
    resize(_buffer.size() == 0 ? 1 : (_buffer.size() * 2));
  }
  push_back_overwrite(value);
}

template <typename T> bool CircularBuffer<T>::push_back_overwrite(const T &value) {
  if (++_tail == _buffer.end()) {
    _tail = _buffer.begin();
  }
  *_tail = value;
  if (_count == _buffer.size()) {
    if (++_head == _buffer.end()) {
      _head = _buffer.begin();
    }
    return false;
  } else {
    if (_count++ == 0) {
      _head = _tail;
    }
    return true;
  }
}

template <typename T> void CircularBuffer<T>::push_back(T &&value) {
  if (!available()) {
    resize(_buffer.size() == 0 ? 1 : (_buffer.size() * 2));
  }
  push_back_overwrite(std::move(value));
}

template <typename T> bool CircularBuffer<T>::push_back_overwrite(T &&value) {
  if (++_tail == _buffer.end()) {
    _tail = _buffer.begin();
  }
  *_tail = std::move(value);
  if (_count == _buffer.size()) {
    if (++_head == _buffer.end()) {
      _head = _buffer.begin();
    }
    return false;
  } else {
    if (_count++ == 0) {
      _head = _tail;
    }
    return true;
  }
}

template <typename T> const T &CircularBuffer<T>::pop_front() {
  if (_count == 0)
    return *_head;
  const T &result = *_head++;
  if (_head == _buffer.end()) {
    _head = _buffer.begin();
  }
  _count--;
  return result;
}

template <typename T> const T &CircularBuffer<T>::pop_back() {
  if (_count == 0)
    return *_tail;
  const T &result = *_tail;
  if (_tail == _buffer.begin()) {
    _tail = _buffer.end();
  }
  _tail--;
  _count--;
  return result;
}

template <typename T> const T &CircularBuffer<T>::first() const {
  return *_head;
}

template <typename T> const T &CircularBuffer<T>::last() const {
  return *_tail;
}

template <typename T>
const T &CircularBuffer<T>::operator[](size_t index) const {
  if (index >= _count)
    return *_tail;
  auto pos = _head - _buffer.begin() + index;
  if (pos >= static_cast<typename std::vector<T>::difference_type>(_buffer.size()))
    pos -= _buffer.size();
  return *(_buffer.begin() + pos);
}

template <typename T> T &CircularBuffer<T>::operator[](size_t index) {
  return const_cast<T &>(
      const_cast<const CircularBuffer<T> *>(this)->operator[](index));
}

template <typename T> size_t CircularBuffer<T>::size() const { return _count; }

template <typename T> size_t CircularBuffer<T>::available() const {
  return _buffer.size() - _count;
}

template <typename T> bool CircularBuffer<T>::isEmpty() const {
  return _count == 0;
}

template <typename T> bool CircularBuffer<T>::isFull() const {
  return _count == _buffer.size();
}

template <typename T> void CircularBuffer<T>::clear() {
  _head = _tail = _buffer.begin();
  _count = 0;
}

template <typename T>
bool CircularBuffer<T>::copyToVector(std::vector<T> &dest) const {
  if (dest.size() < _buffer.size())
    return false;

  if (_head <= _tail) {
    auto head_idx = _head - _buffer.begin();
    std::copy(_buffer.begin() + head_idx, _buffer.begin() + head_idx + _count, dest.begin());
  } else {
    auto head_idx = _head - _buffer.begin();
    auto tail_idx = _tail - _buffer.begin();
    auto it = std::copy(_buffer.begin() + head_idx, _buffer.end(), dest.begin());
    std::copy(_buffer.begin(), _buffer.begin() + tail_idx + 1, it);
  }
  return true;
}

// template <typename T, size_t S>
// template <typename R>
// void CircularBuffer<T, S>::copyToArray(R *dest,
//                                        R (&convertFn)(const T &)) const {
//   const R *finish = dest + count;
//   for (const T *current = head; current < (buffer + capacity) && dest <
//   finish;
//        current++, dest++) {
//     *dest = convertFn(*current);
//   }
//   for (const T *current = buffer; current <= tail && dest < finish;
//        current++, dest++) {
//     *dest = convertFn(*current);
//   }
// }

template <typename T>
bool CircularBuffer<T>::resize(const size_t size) {
  if (size < _count || size == 0)
    return false;
  std::vector<T> tmp(size);
  copyToVector(tmp);
  std::swap(_buffer, tmp);
  _head = _buffer.begin();
  _tail = _count > 0 ? _buffer.begin() + _count - 1 : _buffer.begin();
  return true;
}

template <typename T>
void CircularBuffer<T>::reserve(const size_t capacity) {
  if (capacity <= _buffer.size())
    return;
  // Linearize the buffer if it's wrapped
  if (_head > _tail && _count > 0) {
    std::vector<T> tmp(capacity);
    copyToVector(tmp);
    std::swap(_buffer, tmp);
  } else {
    _buffer.resize(capacity);
  }
  _head = _buffer.begin();
  _tail = _count > 0 ? _buffer.begin() + _count - 1 : _buffer.begin();
}

template <typename T>
void CircularBuffer<T>::shrink_to_fit() {
  if (_count == _buffer.size())
    return;
  std::vector<T> tmp(_count);
  copyToVector(tmp);
  std::swap(_buffer, tmp);
  _head = _buffer.begin();
  _tail = _count > 0 ? _buffer.begin() + _count - 1 : _buffer.begin();
}
} // namespace utils