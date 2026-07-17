import tkinter as tk
import random
import os

PAGES_FILE = "pages.txt"
DEFAULT_PAGES = 5

# ----- 字体设置（在这里统一改大小）-----
FONT_LARGE = ("微软雅黑", 40, "bold")   # 计数器
FONT_MEDIUM = ("微软雅黑", 40)          # 页面指示器
FONT_TEXT = ("微软雅黑", 60)            # 文本编辑区
FONT_BUTTON = ("微软雅黑", 30, "bold")  # 按钮

class RandomSlidesApp:
    def __init__(self, root):
        self.root = root
        self.root.title("随机翻页卡片")
        self.root.geometry("700x600")

        self.pages = []
        self.current_index = 0
        self.click_count = 0

        self.load_pages()

        # ----- 顶部：计数器 + 页面指示 -----
        self.counter_var = tk.StringVar()
        self.update_counter_display()
        counter_label = tk.Label(root, textvariable=self.counter_var,
                                 font=FONT_LARGE, fg="#333")
        counter_label.pack(pady=10)

        self.page_indicator = tk.StringVar()
        indicator = tk.Label(root, textvariable=self.page_indicator,
                             font=FONT_MEDIUM, fg="gray")
        indicator.pack()

        # ----- 底部按钮栏（先打包，固定在窗口底部）-----
        btn_frame = tk.Frame(root)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=15)

        prev_btn = tk.Button(btn_frame, text="◀ 上一页",
                             command=self.prev_page, font=FONT_BUTTON)
        prev_btn.grid(row=0, column=0, padx=8)

        next_btn = tk.Button(btn_frame, text="下一页 ▶",
                             command=self.next_page, font=FONT_BUTTON)
        next_btn.grid(row=0, column=1, padx=8)

        self.random_btn = tk.Button(btn_frame, text="🎲 随机跳转",
                                    command=self.random_jump,
                                    bg="#4CAF50", fg="white",
                                    font=("微软雅黑", 30, "bold"))
        self.random_btn.grid(row=0, column=2, padx=20)

        # ----- 中间文本编辑区（最后打包，填充剩余空间）-----
        self.text_area = tk.Text(root, wrap=tk.WORD, font=FONT_TEXT)
        self.text_area.pack(side=tk.TOP, fill=tk.BOTH, expand=True,
                            padx=20, pady=10)
        self.text_area.bind("<KeyRelease>", self.on_text_edit)

        self.show_page(0)

    # ---------- 以下方法完全不变 ----------
    def load_pages(self):
        if os.path.exists(PAGES_FILE):
            with open(PAGES_FILE, "r", encoding="utf-8") as f:
                content = f.read()
            self.pages = [p.strip() for p in content.split("\n\n") if p.strip()]
            if not self.pages:
                self.pages = [f"这是第 {i+1} 页\n可以在此编辑内容。" for i in range(DEFAULT_PAGES)]
        else:
            self.pages = [f"这是第 {i+1} 页\n可以在此编辑内容。" for i in range(DEFAULT_PAGES)]
            self.save_pages()

    def save_pages(self):
        with open(PAGES_FILE, "w", encoding="utf-8") as f:
            f.write("\n\n".join(self.pages))

    def show_page(self, index):
        if 0 <= index < len(self.pages):
            self.current_index = index
            self.text_area.delete("1.0", tk.END)
            self.text_area.insert("1.0", self.pages[index])
            self.page_indicator.set(f"第 {index+1} 页 / 共 {len(self.pages)} 页")

    def on_text_edit(self, event=None):
        current_text = self.text_area.get("1.0", "end-1c")
        if 0 <= self.current_index < len(self.pages):
            self.pages[self.current_index] = current_text
            self.save_pages()

    def prev_page(self):
        if self.current_index > 0:
            self.show_page(self.current_index - 1)

    def next_page(self):
        if self.current_index < len(self.pages) - 1:
            self.show_page(self.current_index + 1)

    def random_jump(self):
        if len(self.pages) < 2:
            return
        new_index = self.current_index
        while new_index == self.current_index:
            new_index = random.randint(0, len(self.pages) - 1)
        self.show_page(new_index)
        self.click_count += 1
        self.update_counter_display()

    def update_counter_display(self):
        self.counter_var.set(f"点击次数：{self.click_count}")

if __name__ == "__main__":
    root = tk.Tk()
    app = RandomSlidesApp(root)
    root.mainloop()