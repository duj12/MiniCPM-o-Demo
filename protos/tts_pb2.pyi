from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Text(_message.Message):
    __slots__ = ["text", "speaker_id", "secondary_style_id", "is_first", "is_last", "tts_type", "return_sep", "no_padding", "speaker_vector", "line_break", "tts_client_type", "flush_buffer"]
    class SECONDARY_STYLE_ID(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
        jiangpin: _ClassVar[Text.SECONDARY_STYLE_ID]
        cudan: _ClassVar[Text.SECONDARY_STYLE_ID]
    jiangpin: Text.SECONDARY_STYLE_ID
    cudan: Text.SECONDARY_STYLE_ID
    class TTS_TYPE(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
        mltts: _ClassVar[Text.TTS_TYPE]
        hntts: _ClassVar[Text.TTS_TYPE]
        lvtts: _ClassVar[Text.TTS_TYPE]
        entts: _ClassVar[Text.TTS_TYPE]
        jimtts: _ClassVar[Text.TTS_TYPE]
        lyhtts: _ClassVar[Text.TTS_TYPE]
        cn_vc0: _ClassVar[Text.TTS_TYPE]
        en_vc0: _ClassVar[Text.TTS_TYPE]
        entts_vc: _ClassVar[Text.TTS_TYPE]
        AZURE: _ClassVar[Text.TTS_TYPE]
        volcano: _ClassVar[Text.TTS_TYPE]
        eleven_labs: _ClassVar[Text.TTS_TYPE]
    mltts: Text.TTS_TYPE
    hntts: Text.TTS_TYPE
    lvtts: Text.TTS_TYPE
    entts: Text.TTS_TYPE
    jimtts: Text.TTS_TYPE
    lyhtts: Text.TTS_TYPE
    cn_vc0: Text.TTS_TYPE
    en_vc0: Text.TTS_TYPE
    entts_vc: Text.TTS_TYPE
    AZURE: Text.TTS_TYPE
    volcano: Text.TTS_TYPE
    eleven_labs: Text.TTS_TYPE
    TEXT_FIELD_NUMBER: _ClassVar[int]
    SPEAKER_ID_FIELD_NUMBER: _ClassVar[int]
    SECONDARY_STYLE_ID_FIELD_NUMBER: _ClassVar[int]
    IS_FIRST_FIELD_NUMBER: _ClassVar[int]
    IS_LAST_FIELD_NUMBER: _ClassVar[int]
    TTS_TYPE_FIELD_NUMBER: _ClassVar[int]
    RETURN_SEP_FIELD_NUMBER: _ClassVar[int]
    NO_PADDING_FIELD_NUMBER: _ClassVar[int]
    SPEAKER_VECTOR_FIELD_NUMBER: _ClassVar[int]
    LINE_BREAK_FIELD_NUMBER: _ClassVar[int]
    TTS_CLIENT_TYPE_FIELD_NUMBER: _ClassVar[int]
    FLUSH_BUFFER_FIELD_NUMBER: _ClassVar[int]
    text: str
    speaker_id: str
    secondary_style_id: Text.SECONDARY_STYLE_ID
    is_first: bool
    is_last: bool
    tts_type: Text.TTS_TYPE
    return_sep: bool
    no_padding: bool
    speaker_vector: bytes
    line_break: bool
    tts_client_type: str
    flush_buffer: bool
    def __init__(self, text: _Optional[str] = ..., speaker_id: _Optional[str] = ..., secondary_style_id: _Optional[_Union[Text.SECONDARY_STYLE_ID, str]] = ..., is_first: bool = ..., is_last: bool = ..., tts_type: _Optional[_Union[Text.TTS_TYPE, str]] = ..., return_sep: bool = ..., no_padding: bool = ..., speaker_vector: _Optional[bytes] = ..., line_break: bool = ..., tts_client_type: _Optional[str] = ..., flush_buffer: bool = ...) -> None: ...

class InferenceResult(_message.Message):
    __slots__ = ["data_type", "data", "start_time", "end_time", "sentence_index", "char_index", "inference_end", "flush_buffer"]
    class DataType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
        AUDIO: _ClassVar[InferenceResult.DataType]
        CHAR_TIME_MAP: _ClassVar[InferenceResult.DataType]
    AUDIO: InferenceResult.DataType
    CHAR_TIME_MAP: InferenceResult.DataType
    DATA_TYPE_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    START_TIME_FIELD_NUMBER: _ClassVar[int]
    END_TIME_FIELD_NUMBER: _ClassVar[int]
    SENTENCE_INDEX_FIELD_NUMBER: _ClassVar[int]
    CHAR_INDEX_FIELD_NUMBER: _ClassVar[int]
    INFERENCE_END_FIELD_NUMBER: _ClassVar[int]
    FLUSH_BUFFER_FIELD_NUMBER: _ClassVar[int]
    data_type: InferenceResult.DataType
    data: bytes
    start_time: float
    end_time: float
    sentence_index: int
    char_index: int
    inference_end: bool
    flush_buffer: bool
    def __init__(self, data_type: _Optional[_Union[InferenceResult.DataType, str]] = ..., data: _Optional[bytes] = ..., start_time: _Optional[float] = ..., end_time: _Optional[float] = ..., sentence_index: _Optional[int] = ..., char_index: _Optional[int] = ..., inference_end: bool = ..., flush_buffer: bool = ...) -> None: ...

class DurationResult(_message.Message):
    __slots__ = ["data", "start_time", "end_time"]
    DATA_FIELD_NUMBER: _ClassVar[int]
    START_TIME_FIELD_NUMBER: _ClassVar[int]
    END_TIME_FIELD_NUMBER: _ClassVar[int]
    data: str
    start_time: float
    end_time: float
    def __init__(self, data: _Optional[str] = ..., start_time: _Optional[float] = ..., end_time: _Optional[float] = ...) -> None: ...

class DurationInferenceResult(_message.Message):
    __slots__ = ["results", "walk_data"]
    RESULTS_FIELD_NUMBER: _ClassVar[int]
    WALK_DATA_FIELD_NUMBER: _ClassVar[int]
    results: _containers.RepeatedCompositeFieldContainer[DurationResult]
    walk_data: str
    def __init__(self, results: _Optional[_Iterable[_Union[DurationResult, _Mapping]]] = ..., walk_data: _Optional[str] = ...) -> None: ...

class SpeakerInfo(_message.Message):
    __slots__ = ["tts_type", "speaker_path"]
    TTS_TYPE_FIELD_NUMBER: _ClassVar[int]
    SPEAKER_PATH_FIELD_NUMBER: _ClassVar[int]
    tts_type: Text.TTS_TYPE
    speaker_path: str
    def __init__(self, tts_type: _Optional[_Union[Text.TTS_TYPE, str]] = ..., speaker_path: _Optional[str] = ...) -> None: ...

class SpeakerVector(_message.Message):
    __slots__ = ["data", "version"]
    DATA_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    data: bytes
    version: str
    def __init__(self, data: _Optional[bytes] = ..., version: _Optional[str] = ...) -> None: ...

class GetVersionRequest(_message.Message):
    __slots__ = ["tts_type"]
    TTS_TYPE_FIELD_NUMBER: _ClassVar[int]
    tts_type: Text.TTS_TYPE
    def __init__(self, tts_type: _Optional[_Union[Text.TTS_TYPE, str]] = ...) -> None: ...

class Version(_message.Message):
    __slots__ = ["version"]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    version: str
    def __init__(self, version: _Optional[str] = ...) -> None: ...

class SplitTextResult(_message.Message):
    __slots__ = ["texts"]
    TEXTS_FIELD_NUMBER: _ClassVar[int]
    texts: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, texts: _Optional[_Iterable[str]] = ...) -> None: ...

class CheckInputTextResult(_message.Message):
    __slots__ = ["result", "reason"]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    result: bool
    reason: str
    def __init__(self, result: bool = ..., reason: _Optional[str] = ...) -> None: ...
