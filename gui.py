"""Simple desktop launcher. Run: python gui.py"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
import webbrowser

ROOT = Path(__file__).resolve().parent


def resolve_path(value: str, suffix: str, must_exist: bool = False) -> Path:
    if not value.strip():
        raise ValueError('Please fill in every file path.')
    path = Path(value.strip()).expanduser()
    path = (ROOT / path).resolve()
    if path.suffix.lower() != suffix:
        raise ValueError(f'{path.name} must end in {suffix}.')
    if must_exist and not path.is_file():
        raise ValueError(f'Input file does not exist: {path}')
    if not must_exist and path.exists():
        raise ValueError(f'Path already exists: {path}\nPlease choose a new filename.')
    return path


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('PubMed Dataset and Visualization')
        self.geometry('880x820')
        self.minsize(720, 700)
        self.events = queue.Queue()
        self.process = None
        self.running = False
        self.stop_requested = threading.Event()
        self.quit_requested = False
        self.last_plot = None
        self.buttons = []
        self.protocol('WM_DELETE_WINDOW', self.close)
        tabs = ttk.Notebook(self)
        tabs.pack(fill='x', padx=12, pady=12)
        dataset = ttk.Frame(tabs, padding=12)
        embed = ttk.Frame(tabs, padding=12)
        tabs.add(dataset, text='1. Generate dataset')
        tabs.add(embed, text='2. Embed and visualize')
        self.dataset_tab, self.embed_tab, self.tabs = dataset, embed, tabs
        for frame in (dataset, embed):
            frame.columnconfigure(1, weight=1)
        self.csv = self.field(dataset, 0, 'New dataset CSV', 'data/pubmed_run.csv', '.csv')
        ttk.Label(dataset, text='PubMed query').grid(row=1, column=0, sticky='nw', pady=5)
        self.query = ScrolledText(dataset, height=6, wrap='word')
        self.query.grid(row=1, column=1, columnspan=2, sticky='ew', pady=5)
        self.query.insert('1.0', 'pancreatic cancer')
        self.start = self.field(dataset, 2, 'Start date (YYYY/MM/DD)', '2006/01/01')
        self.end = self.field(dataset, 3, 'End date (YYYY/MM/DD)', datetime.now().strftime('%Y/%m/%d'))
        self.limit = self.field(dataset, 4, 'Maximum papers', '15000')
        ttk.Label(dataset, text='NCBI_EMAIL and optional NCBI_API_KEY are read from your .env file.').grid(row=5, column=0, columnspan=3, sticky='w', pady=8)
        self.run_button(dataset, 6, self.generate)
        self.input_csv = self.field(embed, 0, 'Input dataset CSV', 'data/pubmed_run.csv', '.csv', True)
        self.npy = self.field(embed, 1, 'New embeddings NPY', 'outputs/embeddings_run.npy', '.npy')
        self.html = self.field(embed, 2, 'New plot HTML', 'plots/plot_run.html', '.html')
        self.output_csv = self.field(embed, 3, 'New embedded CSV', 'outputs/embedded_run.csv', '.csv')
        self.terms = self.field(embed, 4, 'Global terms to remove', '100')
        self.model = self.field(embed, 5, 'Model', 'sentence-transformers/allenai-specter')
        ttk.Label(embed, text='Clustering').grid(row=6, column=0, sticky='w', pady=5)
        self.cluster = tk.StringVar(value='gmm')
        ttk.Combobox(embed, textvariable=self.cluster, values=('gmm', 'hdbscan', 'none'), state='readonly').grid(row=6, column=1, sticky='ew', pady=5)
        ttk.Label(embed, text='Choose new output filenames for each run. Relative paths start at the repository folder.').grid(row=7, column=0, columnspan=3, sticky='w', pady=8)
        self.run_button(embed, 8, self.visualize)
        self.open_button = ttk.Button(embed, text='Open completed plot', command=self.open_plot, state='disabled')
        self.open_button.grid(row=8, column=2, pady=8)
        controls = ttk.Frame(self)
        controls.pack(fill='x', padx=12, pady=(0, 8))
        self.stop_button = ttk.Button(controls, text='Stop run', command=self.stop_run, state='disabled')
        self.stop_button.pack(side='left')
        ttk.Button(controls, text='Close Program', command=self.close).pack(side='right')
        self.status = tk.StringVar(value='Ready')
        ttk.Label(self, textvariable=self.status).pack(anchor='w', padx=12)
        self.progress = ttk.Progressbar(self, mode='indeterminate')
        self.progress.pack(fill='x', padx=12, pady=6)
        self.log = ScrolledText(self, height=12, state='disabled', wrap='word')
        self.log.pack(fill='both', expand=True, padx=12, pady=(0, 12))
        self.after(100, self.poll)

    def field(self, frame, row, label, default, suffix=None, existing=False):
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky='w', padx=(0, 12), pady=5)
        value = tk.StringVar(value=default)
        ttk.Entry(frame, textvariable=value).grid(row=row, column=1, sticky='ew', pady=5)
        if suffix:
            def browse():
                options = dict(parent=self, initialdir=str(ROOT), filetypes=[(suffix + ' files', '*' + suffix)])
                path = filedialog.askopenfilename(**options) if existing else filedialog.asksaveasfilename(**options, defaultextension=suffix)
                if path:
                    value.set(path)
            ttk.Button(frame, text='Browse...', command=browse).grid(row=row, column=2, padx=(8, 0))
        return value

    def run_button(self, frame, row, callback):
        button = ttk.Button(frame, text='Run', command=callback)
        button.grid(row=row, column=1, sticky='w', pady=8)
        self.buttons.append(button)

    def generate(self):
        try:
            output = resolve_path(self.csv.get(), '.csv')
            query = self.query.get('1.0', 'end').strip()
            if not query:
                raise ValueError('Enter a PubMed query.')
            start, end = self.start.get().strip(), self.end.get().strip()
            if datetime.strptime(start, '%Y/%m/%d') > datetime.strptime(end, '%Y/%m/%d'):
                raise ValueError('Start date must be on or before end date.')
            limit = int(self.limit.get())
            if limit <= 0:
                raise ValueError('Maximum papers must be positive.')
            self.launch('generate_dataset.py', ['--out', str(output), '--term', query, '--start-date', start, '--end-date', end, '--retmax', str(limit), '--no-overwrite'], ('dataset', output))
        except ValueError as exc:
            messagebox.showerror('Check inputs', str(exc), parent=self)

    def visualize(self):
        try:
            inp = resolve_path(self.input_csv.get(), '.csv', True)
            npy = resolve_path(self.npy.get(), '.npy')
            html = resolve_path(self.html.get(), '.html')
            csv = resolve_path(self.output_csv.get(), '.csv')
            if len({inp, npy, html, csv}) != 4:
                raise ValueError('Input and output paths must be different.')
            terms = int(self.terms.get())
            if terms < 0:
                raise ValueError('Global terms to remove must be zero or greater.')
            model = self.model.get().strip()
            if not model:
                raise ValueError('Enter a model name.')
            self.launch('embed_viz.py', ['--in', str(inp), '--embeddings-path', str(npy), '--plot-path', str(html), '--embeddings-csv', str(csv), '--global-terms-to-remove', str(terms), '--model', model, '--cluster', self.cluster.get()], ('plot', html))
        except ValueError as exc:
            messagebox.showerror('Check inputs', str(exc), parent=self)

    def launch(self, script, args, result):
        if self.running:
            return
        self.running = True
        self.process = None
        self.stop_requested.clear()
        self.stop_button.configure(state='normal')
        self.status.set('Running ' + script + '...')
        for button in self.buttons:
            button.configure(state='disabled')
        self.open_button.configure(state='disabled')
        self.log.configure(state='normal')
        self.log.delete('1.0', 'end')
        self.log.configure(state='disabled')
        self.progress.start()
        def worker():
            try:
                self.process = subprocess.Popen([sys.executable, '-u', str(ROOT / script), *args], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                if self.stop_requested.is_set():
                    self.process.kill()
                for line in self.process.stdout:
                    self.events.put(('log', line))
                code = self.process.wait()
                self.process.stdout.close()
                self.events.put(('done', (code, result)))
            except Exception as exc:
                self.events.put(('log', str(exc) + '\n'))
                self.events.put(('done', (1, result)))
        threading.Thread(target=worker, daemon=True).start()

    def poll(self):
        for _ in range(300):
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == 'log':
                self.log.configure(state='normal')
                self.log.insert('end', payload)
                self.log.see('end')
                self.log.configure(state='disabled')
            else:
                code, (kind, path) = payload
                self.running = False
                self.process = None
                self.stop_button.configure(state='disabled')
                self.progress.stop()
                for button in self.buttons:
                    button.configure(state='normal')
                if self.quit_requested:
                    self.destroy()
                    return
                if self.stop_requested.is_set():
                    self.status.set('Stopped. Any output files already written may be incomplete.')
                    continue
                self.status.set('Completed: ' + str(path) if code == 0 else 'Run failed. See the log for details.')
                if code != 0:
                    messagebox.showerror('Run failed', 'See the log for details.', parent=self)
                elif kind == 'dataset':
                    self.input_csv.set(str(path))
                    self.tabs.select(self.embed_tab)
                else:
                    self.last_plot = path
                    self.open_button.configure(state='normal')
        self.after(100, self.poll)

    def open_plot(self):
        if self.last_plot:
            webbrowser.open(self.last_plot.as_uri())

    def stop_run(self):
        if not self.running:
            return
        self.stop_requested.set()
        self.stop_button.configure(state='disabled')
        self.status.set('Stopping run...')
        process = self.process
        if process is not None and process.poll() is None:
            try:
                process.kill()
            except OSError as exc:
                # It may have exited between poll() and kill().
                if process.poll() is None:
                    self.stop_requested.clear()
                    self.stop_button.configure(state='normal')
                    messagebox.showerror('Could not stop run', str(exc), parent=self)

    def close(self):
        if self.running:
            if not messagebox.askyesno('Stop and quit?',
                    'Force stop the active run and quit? Output files already written may be incomplete.', parent=self):
                return
            self.stop_run()
            if self.stop_requested.is_set():
                self.quit_requested = True
            return
        self.destroy()


if __name__ == '__main__':
    App().mainloop()
