import gradio as gr
import requests


def generate_outfit(prompt: str, seed: int):
    resp = requests.post(
        "http://127.0.0.1:14891/generate",
        json={
            "prompt": prompt,
            "height": 1024,
            "width": 1024,
            "num_images": 1,
            "guidance_scale": 4.5,
            "num_inference_steps": 40,
            "seed": seed,
        },
    )
    resp.raise_for_status()

    data = resp.json()

    image_paths = data["images"]

    print("image_paths", image_paths)

    return image_paths[0]


def main():
    with gr.Blocks() as app:
        gr.Markdown("# AI Outfit Generator\n\nby Yuxin.Nie")

        prompt = gr.Textbox(lines=5, label="Prompt", placeholder="Write a prompt here")

        seed = gr.Number(1234, label="Seed", minimum=0, maximum=60000)

        btn = gr.Button("Generate Outfit")

        generated_image = gr.Image(label="Generated Image")

        btn.click(fn=generate_outfit, inputs=[prompt, seed], outputs=[generated_image])

    app.launch(server_name="0.0.0.0", server_port=14892)


if __name__ == "__main__":
    main()
