# built-in libraries
from typing import TypeVar, Any, List, Optional, Type, Union

# litellm libraries
import litellm
from litellm.types.utils import ModelResponse, Message
from litellm import completion
from instructor import from_litellm, Mode

# misc libraries
from pydantic import BaseModel, create_model


class BaseResponse(BaseModel):
    """A default response model that defines a single
    field `answer` to store the response from the LLM."""
    answer: str


# For type-hinting structured responses:
ResponseType = TypeVar("ResponseType", bound=BaseModel)


class LLMCaller:
    """A class to interact with an LLM using LiteLLM and Instructor."""

    def __init__(
        self,
        api_key: str,
        model_id: str,
        project_id: Optional[str] = None,
        api_url: Optional[str] = None,
        params: dict[str, Any] = {},
        mode: Mode = Mode.JSON
    ):
        self.api_key = api_key
        self.project_id = project_id
        self.api_url = api_url
        self.model_id = model_id
        self.params = params

        # Boilerplate for Watsonx.ai:
        litellm.drop_params = True
        # Instructor client for Pydantic-based interactions:
        self.client = from_litellm(completion, mode=mode)

    def create_response_model(self, title: str, fields: dict) -> ResponseType:
        """Dynamically create a Pydantic model inheriting from BaseResponse."""
        return create_model(title, **fields, __base__=BaseResponse)  # type: ignore

    def invoke(
        self,
        messages: List[Message],
        response_model: Optional[Type[ResponseType]] = BaseResponse,
        **kwargs
    ) -> Union[ResponseType, str]:
        # Prepare call arguments, only include optional fields if provided
        call_args: dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "api_key": self.api_key,
            **kwargs,
        }
        if self.project_id:
            call_args["project_id"] = self.project_id
        if self.api_url:
            call_args["api_base"] = self.api_url

        if response_model is None:
            # Raw-text path
            resp: ModelResponse = completion(**call_args)  # type: ignore
            # Extract the first choice's content
            return resp.choices[0].message.content

        # Structured path
        resp = self.client.chat.completions.create(
            response_model=response_model,
            **call_args  # type: ignore
        )
        return resp  # already parsed into a BaseModel subclass

    def chat(
        self,
        messages: List[Union[dict[str, str], Message]],
        **kwargs
    ) -> ModelResponse:
        """
        Another legacy path: calls the underlying `completion` and returns
        the full ModelResponse object.
        """
        call_args: dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "api_key": self.api_key,
            **kwargs,
        }
        if self.project_id:
            call_args["project_id"] = self.project_id
        if self.api_url:
            call_args["api_base"] = self.api_url

        return completion(**call_args)
