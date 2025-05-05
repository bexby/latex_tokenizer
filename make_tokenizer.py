from transformers import AutoTokenizer, AutoModel

model_name = "sentence-transformers/all-distilroberta-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

latex_tokens = [
    # 基本运算符
    "\\frac", "\\sqrt", "\\cdot", "\\times", "\\div", "\\pm", "\\mp", "\\ast", "\\star",

    # 关系符
    "\\leq", "\\geq", "\\neq", "\\approx", "\\equiv", "\\sim", "\\propto",

    # 箭头符号
    "\\rightarrow", "\\leftarrow", "\\Rightarrow", "\\Leftarrow", "\\leftrightarrow",

    # 极限与积分
    "\\sum", "\\prod", "\\int", "\\lim", "\\infty", "\\partial", "\\nabla", "\\oint",

    # 集合与逻辑
    "\\in", "\\notin", "\\subset", "\\supset", "\\subseteq", "\\supseteq",
    "\\cup", "\\cap", "\\exists", "\\forall", "\\neg", "\\land", "\\lor",

    # 函数与运算符
    "\\sin", "\\cos", "\\tan", "\\csc", "\\sec", "\\cot",
    "\\log", "\\ln", "\\exp",

    # 希腊字母（小写）
    "\\alpha", "\\beta", "\\gamma", "\\delta", "\\epsilon", "\\zeta", "\\eta",
    "\\theta", "\\iota", "\\kappa", "\\lambda", "\\mu", "\\nu", "\\xi",
    "\\pi", "\\rho", "\\sigma", "\\tau", "\\upsilon", "\\phi", "\\chi", "\\psi", "\\omega",

    # 希腊字母（大写）
    "\\Gamma", "\\Delta", "\\Theta", "\\Lambda", "\\Xi", "\\Pi", "\\Sigma", "\\Phi", "\\Psi", "\\Omega",

    # 括号
    "\\left(", "\\right)", "\\left[", "\\right]", "\\left\\{", "\\right\\}",

    # 其他
    "\\dots", "\\ldots", "\\cdots", "\\vdots", "\\ddots",
    "\\text", "\\mathrm", "\\mathbb", "\\mathbf", "\\mathcal",
]

num_added = tokenizer.add_tokens(latex_tokens)
print(f"共添加了 {num_added} 个 LaTeX token")

model.resize_token_embeddings(len(tokenizer))
print("模型 embedding 已调整为新词表大小。")

tokenizer.save_pretrained("./init_model")
model.save_pretrained("./init_model")