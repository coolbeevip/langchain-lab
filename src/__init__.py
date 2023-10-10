# import os
#
# from dotenv import load_dotenv
#
# WORK_DIR = os.getcwd()
# print(WORK_DIR)
# load_dotenv()
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
#
# if "OPENAI_API_BASE" not in os.environ:
#     os.environ["OPENAI_API_BASE"] = "https://api.openai.com/v1"
#
# if "DEFAULT_AI_PLATFORM_SUPPORT" not in os.environ:
#     os.environ["DEFAULT_AI_PLATFORM_SUPPORT"] = "OpenAI"
#
# if "DEFAULT_AI_PLATFORM" not in os.environ:
#     os.environ["DEFAULT_AI_PLATFORM"] = "OpenAI"
#
# if "HUGGINGFACE_CATCH_PATH" not in os.environ:
#     os.environ["HUGGINGFACE_CATCH_PATH"] = os.path.join(WORK_DIR, "huggingface")
