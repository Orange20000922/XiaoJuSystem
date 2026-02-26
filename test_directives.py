import sys, os, torch, json, io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

project_root = Path("D:/Users/21405/source/repos/MyNeuroLikeSystem/neuro_like_system")
sys.path.insert(0, str(project_root))
os.environ["HF_HUB_OFFLINE"] = "1"

from models.joint_model import create_joint_model
from configs.model_config import JointModelConfigLowVRAM, DEFAULT_PERSONALITY
from configs.config_loader import load_emotion_prompt_config

with open(project_root / "config.json", encoding="utf-8") as f:
    cfg = json.load(f)
ep_cfg = load_emotion_prompt_config(cfg)

config = JointModelConfigLowVRAM()
checkpoint = torch.load(
    project_root / "checkpoints/joint_model/best.pt",
    map_location="cuda", weights_only=False
)
model, tokenizer = create_joint_model(config)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to("cuda").eval()
personality = torch.tensor(DEFAULT_PERSONALITY.to_embedding_vector())

def build_directive(emotion_analysis):
    parts = []
    emotion = emotion_analysis["emotion"]["primary"]
    intensity = emotion_analysis["emotion"]["intensity"]
    bert_prob = emotion_analysis["emotion"]["primary_prob"]
    behavior = emotion_analysis["behavior"]["type"]
    tone = emotion_analysis["behavior"]["tone"]

    reliability = ep_cfg.emotion_reliability.get(emotion, 0.7)
    eff = bert_prob * reliability
    strong = ep_cfg.confidence_thresholds["strong"]
    weak = ep_cfg.confidence_thresholds["weak"]

    emotion_directive = ep_cfg.emotion_map.get(emotion, "")
    if emotion_directive:
        if eff >= strong:
            if intensity >= ep_cfg.intensity_levels.get("high_min", 0.7):
                emotion_directive = f"（强烈）{emotion_directive}"
            parts.append(emotion_directive)
        elif eff >= weak:
            stripped = emotion_directive.lstrip("用户")
            parts.append(f"用户可能{stripped}，但不确定，结合上下文判断")

    return "。".join(parts) if parts else "(无指令)"

test_cases = [
    "哈哈哈哈笑死我了",
    "泪目了",
    "卧槽！！！",
    "恶心",
    "气死我了",
    "好可怕",
    "这是什么？",
    "666666",
    "好温柔啊",
    "好无聊啊",
    "大家好",
    "太恶心了吧",
    "怕了怕了",
    "你们觉得呢",
    "草 太好笑了吧",
    "加油加油",
    "我裂开了",
    "真下头",
    "awsl",
    "前方高能",
    "这也太离谱了",
]

print(f"{'文本':<12} {'情绪':>10} {'prob':>5} {'rel':>5} {'eff':>5}  指令")
print("=" * 100)
for text in test_cases:
    result = model.predict(text, personality, tokenizer, device="cuda")
    e = result["emotion"]["primary"]
    p = result["emotion"]["primary_prob"]
    rel = ep_cfg.emotion_reliability.get(e, 0.7)
    eff = p * rel
    directive = build_directive(result)
    print(f"{text:<12} {e:>10} {p:5.2f} {rel:5.2f} {eff:5.3f}  {directive}")

