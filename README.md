# How to start

This package includes scripts to download the pre-trained model from huggingface, and convert the model to GGUF format as well.

Therefore, you are required to install `huggingface-cli` and log in with a huggingface account.

You can use homebrew to install the `huggingface-cli`:
```
brew install huggingface-cli
```

More info about `huggingface-cli`, you can visit the page: [https://huggingface.co/docs/huggingface_hub/guides/cli](https://huggingface.co/docs/huggingface_hub/guides/cli)

## Build models

To start building the models, run the following command:

```
yarn make_convertor  # Build the model convertor
yarn download  # Download models from huggingface
yarn convert  # Convert the models to GGUF format
```

## Startup test server

This project contains a simple chat server. You can test the model in local machine.

To begin with test server, you need to startup docker and run the follow script:

```
./scripts/test
```
