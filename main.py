from diffusers import DiffusionPipeline
import torch
import matplotlib.pyplot as plt


def main():
    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    pipe.to("cuda")

    prompt = """
Generate an image of nasi lemak ayam, with these ingredients:
1. Chicken drumsticks.
2. Sambal.
3. Beans.
4. Ikan teri.
5. Egg.
    """
    image = pipe(prompt).images[0]

    # create figure
    fig = plt.figure(figsize=(10, 7))

    rows, columns = 2,2

    # Adds a subplot at the 1st position
    fig.add_subplot(rows, columns, 1)

    # Showing the image
    plt.imshow(image)
    plt.axis('off')
    plt.title("image")


    # Saving Image locally
    image.save("image.jpg")


if __name__ == "__main__":
    main()
