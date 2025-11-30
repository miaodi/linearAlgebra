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
CircularBuffer<T>::CircularBuffer(const size_t S)
    : _buffer(make_buffer(S)), _capacity(S),
      _head(_buffer.get()), _tail(_buffer.get()), _count(0)
{
}

template <typename T> void CircularBuffer<T>::push_front(const T &value) {
  if (!available()) {
    resize(_capacity == 0 ? 1 : (_capacity * 2));
  }
  push_front_overwrite(value);
}

template <typename T> bool CircularBuffer<T>::push_front_overwrite(const T &value) {
  if (_capacity == 0)
    return false;
  auto begin = _buffer.get();
  auto end = begin + _capacity;
  if (_head == begin) {
    _head = end;
  }
  *--_head = value;
  if (_count == _capacity) {
    if (_tail == begin) {
      _tail = end;
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
    resize(_capacity == 0 ? 1 : (_capacity * 2));
  }
  push_front_overwrite(std::move(value));
}

template <typename T> bool CircularBuffer<T>::push_front_overwrite(T &&value) {
  if (_capacity == 0)
    return false;
  auto begin = _buffer.get();
  auto end = begin + _capacity;
  if (_head == begin) {
    _head = end;
  }
  *--_head = std::move(value);
  if (_count == _capacity) {
    if (_tail == begin) {
      _tail = end;
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
    resize(_capacity == 0 ? 1 : (_capacity * 2));
  }
  push_back_overwrite(value);
}

template <typename T> bool CircularBuffer<T>::push_back_overwrite(const T &value) {
  if (_capacity == 0)
    return false;
  auto begin = _buffer.get();
  auto end = begin + _capacity;
  if (++_tail == end) {
    _tail = begin;
  }
  *_tail = value;
  if (_count == _capacity) {
    if (++_head == end) {
      _head = begin;
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
    resize(_capacity == 0 ? 1 : (_capacity * 2));
  }
  push_back_overwrite(std::move(value));
}

template <typename T> bool CircularBuffer<T>::push_back_overwrite(T &&value) {
  if (_capacity == 0)
    return false;
  auto begin = _buffer.get();
  auto end = begin + _capacity;
  if (++_tail == end) {
    _tail = begin;
  }
  *_tail = std::move(value);
  if (_count == _capacity) {
    if (++_head == end) {
      _head = begin;
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
  auto begin = _buffer.get();
  auto end = begin + _capacity;
  const T &result = *_head++;
  if (_head == end) {
    _head = begin;
  }
  _count--;
  return result;
}

template <typename T> const T &CircularBuffer<T>::pop_back() {
  if (_count == 0)
    return *_tail;
  auto begin = _buffer.get();
  auto end = begin + _capacity;
  const T &result = *_tail;
  if (_tail == begin) {
    _tail = end;
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
  auto begin = _buffer.get();
  auto pos = static_cast<size_t>(_head - begin) + index;
  if (pos >= _capacity)
    pos -= _capacity;
  return *(begin + pos);
}

template <typename T> T &CircularBuffer<T>::operator[](size_t index) {
  return const_cast<T &>(
      const_cast<const CircularBuffer<T> *>(this)->operator[](index));
}

template <typename T> size_t CircularBuffer<T>::size() const { return _count; }

template <typename T> size_t CircularBuffer<T>::available() const {
  return _capacity - _count;
}

template <typename T> bool CircularBuffer<T>::empty() const {
  return _count == 0;
}

template <typename T> bool CircularBuffer<T>::full() const {
  return _count == _capacity;
}

template <typename T> void CircularBuffer<T>::clear() {
  _head = _tail = _buffer.get();
  _count = 0;
}

template <typename T>
void CircularBuffer<T>::copy_to_array(T *dest) const {
  if (_count == 0)
    return;

  auto begin = _buffer.get();
  auto end = begin + _capacity;

  if (_head <= _tail) {
    std::copy(_head, _head + _count, dest);
  } else {
    auto head_count = static_cast<size_t>(end - _head);
    auto it = std::copy(_head, end, dest);
    std::copy(begin, begin + (_count - head_count), it);
  }
}

template <typename T>
bool CircularBuffer<T>::dump_to_vector(std::vector<T>& dest) const
{
    if (dest.size() < _count)
        return false;
    copy_to_array(dest.data());
    return true;
}

template <typename T>
bool CircularBuffer<T>::resize(const size_t size) {
  if (size < _count || size == 0)
    return false;
  auto tmp = make_buffer(size);
  copy_to_array(tmp.get());
  _buffer.swap(tmp);
  _capacity = size;
  _head = _buffer.get();
  _tail = _count > 0 ? _head + _count - 1 : _head;
  return true;
}

template <typename T>
void CircularBuffer<T>::reserve(const size_t capacity)
{
    if (capacity <= _capacity)
        return;
    _buffer.swap(make_buffer(capacity));
    _capacity = capacity;
    _head = _buffer.get();
    _tail = _head;
    _count = 0;
}

template <typename T>
void CircularBuffer<T>::shrink_to_fit() {
  if (_count == _capacity)
    return;
  auto tmp = make_buffer(_count);
  copy_to_array(tmp.get());
  _buffer.swap(tmp);
  _capacity = _count;
  _head = _buffer.get();
  _tail = _count > 0 ? _head + _count - 1 : _head;
}
} // namespace utils
