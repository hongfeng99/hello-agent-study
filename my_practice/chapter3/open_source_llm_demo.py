import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    model_id = "Qwen/Qwen1.5-0.5B-Chat"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Using device: {device}")
    print("正在加载 tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print("正在加载模型，第一次运行会自动下载模型文件，可能需要几分钟...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype
    ).to(device)

    model.eval()

    messages = [
        {"role": "system", "content": "你是一个耐心、清晰的人工智能学习助手。"},
        {"role": "user", "content": "你好，请用三句话介绍一下你自己。"}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    print("开始生成回答...")

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            repetition_penalty=1.05
        )

    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    print("\n模型回答：")
    print(response)


if __name__ == "__main__":
    main()