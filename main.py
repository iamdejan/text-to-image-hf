import os
from huggingface_hub import InferenceClient


def main():
    try:
        prompt = """
Generate an image of nasi lemak ayam, with these ingredients:
1. Chicken drumsticks.
2. Sambal.
3. Beans.
4. Ikan teri.
5. Egg.
    """
        print("Generating image, please wait...")

        client = InferenceClient(
            provider="fal-ai",
            api_key=os.environ["HF_TOKEN"],
        )

        # output is a PIL.Image object
        image = client.text_to_image(
            prompt,
            model="Tongyi-MAI/Z-Image-Turbo",
        )
        image.save("generated_flux_image.png")
        image.show()

    except Exception as e:
        print(f"Failed to generate image: {e}")


if __name__ == "__main__":
    main()
