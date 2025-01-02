
import os
import argparse

from .. import io


now_number = 0
def web_data_view():
    import gradio as gr
    global now_number
    parser = argparse.ArgumentParser()
    parser.usage = "vvcli web [OPTIONS]"
    parser.add_argument('--file_path', type=str, required=True, help="Path to the json file")
    args = parser.parse_args()


    if not os.path.exists(args.file_path):
        raise FileNotFoundError(f"File not found: {args.file_path}.")
    if not os.path.isfile(args.file_path):
        raise FileNotFoundError(f"File is a directory: {args.file_path}.")
    data = io.jsonload(args.file_path)
    props = list(data[0].keys())
    with gr.Blocks() as demo:
        with gr.Row():
            pre_btn = gr.Button("<<")
            number = gr.Number(value=now_number, label="index")
            jump_btn = gr.Button("Jump")
            next_btn = gr.Button(">>")
        props_boxes = []
        for prop in props:
            props_boxes.append(gr.Textbox(label=prop, value=data[0][prop]))

        @pre_btn.click(inputs=[], outputs=[number] + props_boxes)
        def pre():
            global now_number
            if now_number == 0:
                gr.Warning("Already at the first item.")
            else:
                now_number -= 1
            return [now_number] + [str(data[now_number][prop]) for prop in props]

        @next_btn.click(inputs=[], outputs=[number] + props_boxes)
        def next():
            global now_number
            if now_number == len(data) - 1:
                gr.Warning("Already at the last item.")
            else:
                now_number += 1
            return [now_number] + [str(data[now_number][prop]) for prop in props]

        @jump_btn.click(inputs=[number], outputs=[number] + props_boxes)
        def next(n):
            global now_number
            if n < 0:
                gr.Warning("n is set to 0")
                n = 0
            elif n >= len(data):
                gr.Warning(f"n is set to {len(data) - 1}")
                n = len(data) - 1
            now_number = n
            return [n] + [str(data[n][prop]) for prop in props]
    demo.queue().launch()
