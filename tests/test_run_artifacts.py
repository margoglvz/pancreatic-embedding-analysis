"""Offline tests for shared CLI/GUI run artifact behavior."""
import argparse
import ast
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import plotly.express as px
import generate_dataset as dataset
import run_artifacts as runs


class RunTests(unittest.TestCase):
    def test_dataset_and_visualization_share_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            argv = ['generate_dataset.py', '--runs-dir', tmp, '--term', 'test query', '--email', 'private@example.com', '--api-key', 'secret', '--out', 'papers.csv']
            article = {'MedlineCitation': {'PMID': '1', 'Article': {'ArticleTitle': 'alpha', 'Abstract': {'AbstractText': ['text']}, 'Journal': {'Title': 'journal'}}}}
            with patch.object(sys, 'argv', argv), patch.object(dataset, 'gather_pmids_time_slice', return_value=['1']), patch.object(dataset, 'fetch_records', return_value=[article]):
                dataset.main()
            run, = root.iterdir()
            config = runs.read_config(run)
            self.assertEqual(config['stages']['generate_dataset']['status'], 'completed')
            self.assertNotIn('secret', (run/'config.json').read_text())
            self.assertNotIn('private@example.com', (run/'config.json').read_text())
            self.assertEqual(runs.visualization_run(run/'papers.csv', root=root), run)
            tree = ast.parse(Path('embed_viz.py').read_text(encoding='utf-8'))
            main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
            ns = dict(argparse=argparse, os=os, np=np, pd=pd, px=px,
                      visualization_run=runs.visualization_run, copy_input=runs.copy_input, artifact_name=runs.artifact_name, stage=runs.stage,
                      embed_texts=Mock(return_value=np.ones((1, 3))), reduce_umap=Mock(return_value=np.array([[0., 0.]])),
                      cluster_top_terms_tfidf=Mock(return_value={0: ['alpha']}), get_top_cancer_terms=Mock(return_value=({}, {})),
                      pick_two_largest_clusters=Mock(return_value=None), make_cluster_hover_text=Mock(return_value='test'))
            exec(compile(ast.Module(body=[main], type_ignores=[]), 'embed_viz.py', 'exec'), ns)
            with patch.object(sys, 'argv', ['embed_viz.py', '--in', str(run/'papers.csv'), '--cluster', 'none', '--no-kde']):
                ns['main']()
            self.assertEqual({p.name for p in run.iterdir()}, {'papers.csv','embeddings.npy','embedded.csv','plot.html','config.json'})
            config = runs.read_config(run)
            self.assertEqual(config['stages']['embed_viz']['status'], 'completed')
            self.assertEqual(config['stages']['generate_dataset']['settings']['term'], 'test query')
            next_run = runs.visualization_run(run/'papers.csv', root=root)
            self.assertNotEqual(run, next_run)
            copied = runs.copy_input(next_run, run/'papers.csv')
            self.assertEqual(copied.read_bytes(), (run/'papers.csv').read_bytes())

    def test_failed_and_interrupted_status_and_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = runs.create_run(tmp)
            with self.assertRaises(RuntimeError):
                with runs.stage(run, 'example', {'model': 'test'}, {'plot_html': 'plot.html'}):
                    raise RuntimeError('failure')
            self.assertEqual(runs.read_config(run)['stages']['example']['status'], 'failed')
            with self.assertRaises(FileExistsError):
                with runs.stage(run, 'example', {}, {}):
                    pass
            config = runs.read_config(run)
            config['stages']['example']['status'] = 'running'
            runs.write_config(run, config)
            runs.mark_interrupted(run)
            self.assertEqual(runs.read_config(run)['stages']['example']['status'], 'interrupted')
            (run/'taken.csv').write_text('preserve')
            with self.assertRaises(FileExistsError):
                with runs.stage(run, 'generate_dataset', {}, {'pubmed_csv': 'taken.csv'}):
                    pass
            self.assertEqual((run/'taken.csv').read_text(), 'preserve')

    def test_gui_passes_shared_run_directory(self):
        import gui
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            run = runs.create_run(tmp)
            value = lambda text: Mock(get=Mock(return_value=text))
            app = SimpleNamespace(csv=value('papers.csv'), query=value('test query'),
                start=value('2020/01/01'), end=value('2021/01/01'), limit=value('50'), launch=Mock())
            with patch.object(gui, 'create_run', return_value=run):
                gui.App.generate(app)
            args = app.launch.call_args.args[1]
            self.assertEqual(args[args.index('--run-dir')+1], str(run))
            self.assertEqual(app.launch.call_args.args[2], ('dataset', run/'papers.csv'))
            inp = run/'papers.csv'
            inp.write_text('title,abstract\nalpha,beta\n')
            app.input_csv=value(str(inp))
            app.npy=value('embeddings.npy')
            app.html=value('plot.html')
            app.output_csv=value('embedded.csv')
            app.terms=value('100')
            app.model=value('test-model')
            app.cluster=value('gmm')
            with patch.object(gui, 'visualization_run', return_value=run):
                gui.App.visualize(app)
            args = app.launch.call_args.args[1]
            self.assertEqual(args[args.index('--run-dir')+1], str(run))
            self.assertEqual(app.launch.call_args.args[2], ('plot', run/'plot.html'))

    def test_standalone_input_and_unique_folders(self):
        with tempfile.TemporaryDirectory() as tmp:
            inp = Path(tmp)/'source.csv'
            inp.write_text('title,abstract\nalpha,beta\n')
            run = runs.visualization_run(inp, root=Path(tmp)/'runs')
            copy = runs.copy_input(run, inp)
            with runs.stage(run, 'embed_viz', {'inp': str(inp)}, {'pubmed_csv': copy.name}):
                pass
            self.assertEqual(copy.read_bytes(), inp.read_bytes())
            self.assertNotEqual(run, runs.create_run(Path(tmp)/'runs'))


if __name__ == '__main__':
    unittest.main()
