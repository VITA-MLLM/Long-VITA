# coding=utf-8
# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def check_condition(cond):
    if not cond:
        raise RuntimeError


class NotDivisibleError(Exception):
    def __init__(self, denominator, molecule, error_info):
        super().__init__()
        self._error_info = error_info
        self._molecule = molecule
        self._denominator = denominator

    def __str__(self):
        if self._error_info is None:
            return f"{self._denominator} is not divisible by {self._molecule}"
        else:
            return self._error_info.format(self._denominator, self._molecule)


def check_divisible(denominator, molecule, error_info=None):
    if denominator % molecule == 0:
        return
    raise NotDivisibleError(denominator, molecule, error_info)


def check_divisible_by_zero(dividend, divisor):
    if not isinstance(divisor, int) or divisor != 0:
        return dividend / divisor
    raise ZeroDivisionError


class NotEqualError(Exception):
    def __init__(self, tensor_a, tensor_b, error_info):
        super().__init__()
        self._error_info = error_info
        self._tensor_a = tensor_a
        self._tensor_b = tensor_b

    def __str__(self):
        if self._error_info is None:
            return f"{self._tensor_a} is not equal to {self._tensor_b}"
        else:
            return self._error_info.format(self._tensor_a, self._tensor_b)


def check_equal(tensor_a, tensor_b, error_info=None):
    if tensor_a == tensor_b:
        return
    raise NotEqualError(tensor_a, tensor_b, error_info)


class NotExistError(Exception):
    def __init__(self, item, container, error_info):
        super().__init__()
        self._error_info = error_info
        self._item = item
        self._container = container

    def __str__(self):
        if self._error_info is None:
            return f"{self._item} is not in {self._container}"
        else:
            return self._error_info.format(self._item, self._container)


def check_exist(item, container, error_info=None):
    if item in container:
        return True
    raise NotExistError(item, container, error_info)


class NotExpectedTypeError(Exception):
    def __init__(self, variable, expected_type, error_message=None):
        super().__init__()
        self._variable = variable
        self._expected_type = expected_type
        self._error_message = error_message

    def __str__(self):
        if self._error_message:
            return self._error_message
        else:
            return f"expected {self._expected_type.__name__} find {type(self._variable).__name__}"


def check_type(variable, expected_type, error_message=None):
    if isinstance(variable, expected_type):
        return
    else:
        raise NotExpectedTypeError(variable, expected_type, error_message)


class IsNotNoneError(Exception):
    def __init__(self, error_message):
        super().__init__()
        self._error_message = error_message

    def __repr__(self):
        if self._error_message:
            return self._error_message
        else:
            return "expect None variable"


class IsNoneError(Exception):
    def __init__(self, error_message):
        super().__init__()
        self._error_message = error_message

    def __repr__(self):
        if self._error_message:
            return self._error_message
        else:
            return "expect not None variable"


def ensure_var_is_not_none(variable, error_message=None):
    if variable is not None:
        return
    else:
        raise IsNoneError(error_message=error_message)


def ensure_var_is_none(variable, error_message=None):
    if variable is None:
        return
    else:
        raise IsNotNoneError(error_message)


class IsNotValidError(Exception):
    def __init__(self, error_message):
        super().__init__()
        self._error_message = error_message

    def __repr__(self):
        if self._error_message:
            return self._error_message
        else:
            return "Expression is not valid"


def ensure_valid(expression, error_message=None):
    if not expression:
        raise IsNotValidError(error_message)
        
        
class GPTDatasetSampleIndexError(Exception):
    def __init__(self, error_message):
        super().__init__()
        self._error_message = error_message

    def __repr__(self):
        if self._error_message:
            return self._error_message
        else:
            return "Bad sample index."
