import os
from huggingface_hub import InferenceClient


def main():
    try:
        prompt = """
Generate an image of chicken nasi lemak, based on these ingredients:
1. Chicken thighs and drumsticks.
2. Sambal.
3. Beans.
4. Deep fried anchovies.
5. Sunny-side up egg.
    """
        print("Generating image, please wait...")

        client = InferenceClient(
            provider="fal-ai",
            api_key=os.environ["HF_TOKEN"],
        )

        # output is a PIL.Image object
        model = "Tongyi-MAI/Z-Image-Turbo"
        image = client.text_to_image(
            prompt,
            model=model,
        )
        image.save(f"{model}.png")
        image.show()

    except Exception as e:
        print(f"Failed to generate image: {e}")


if __name__ == "__main__":
    main()
