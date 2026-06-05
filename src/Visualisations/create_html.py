#!/usr/bin/env python
import argparse
import json
from pathlib import Path

HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    body {{ margin:0; overflow:hidden; font-family: sans-serif; }}
    #plot {{ width:100vw; height:100vh; }}

    /* Clean text card */
    #hover-card {{
      position: fixed;
      display: none;
      width: 220px;
      padding: 12px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.15);
      border: 1px solid #ddd;
      border-radius: 6px;
      background: #fff;
      pointer-events: none;
      z-index: 9999;
    }}
    .title {{ font-weight: bold; color: #222; margin-bottom: 5px; border-bottom: 1px solid #eee; padding-bottom: 4px; }}
    .sub {{ font-size: 12px; color: #666; word-wrap: break-word; }}
  </style>
</head>
<body>
  <div id="plot"></div>

  <div id="hover-card">
    <div class="title" id="hc-author"></div>
    <div class="sub" id="hc-file"></div>
  </div>

  <script type="application/json" id="tsne-data">{json_data}</script>

  <script>
    // Parse inline JSON
    const raw = document.getElementById('tsne-data').textContent;
    let data = [];
    try {{
      data = JSON.parse(raw);
    }} catch (e) {{
      document.body.innerHTML = '<pre style="color:red">Failed to parse embedded JSON: ' + e + '</pre>';
      throw e;
    }}

    const trace = {{
      x: data.map(d => d.x),
      y: data.map(d => d.y),
      mode: 'markers',
      type: 'scatter',
      text: data.map(d => d.label),
      hoverinfo: 'none', 
      marker: {{
        size: 8,
        color: data.map(d => d.color),
        symbol: data.map(d => d.shape),
        opacity: 0.9
      }}
    }};

    const layout = {{
      title: '{title}',
      xaxis: {{ showgrid: false, zeroline: false }},
      yaxis: {{ showgrid: false, zeroline: false }},
      hovermode: 'closest',
      margin: {{ l:40, r:40, t:60, b:40 }}
    }};

    const config = {{
      toImageButtonOptions: {{
        format: 'png',
        filename: '{download_name}'  
      }}
    }};

    Plotly.newPlot('plot', [trace], layout, config).then(gd => {{
      const card = document.getElementById('hover-card');
      const authorDiv = document.getElementById('hc-author');
      const fileDiv = document.getElementById('hc-file');
      const PAD = 15; 

      function placeCard(clientX, clientY) {{
        const vw = window.innerWidth, vh = window.innerHeight;
        const rect = card.getBoundingClientRect();

        let left = clientX + PAD;
        let top  = clientY + PAD;
        if (left + rect.width > vw) left = clientX - PAD - rect.width;
        if (top  + rect.height > vh) top  = clientY - PAD - rect.height;

        card.style.left = left + 'px';
        card.style.top  = top  + 'px';
      }}

      gd.on('plotly_hover', ev => {{
        const p = ev.points[0];
        const d = data[p.pointIndex];
        if (!d) return;

        let fileName = d.filename ? d.filename : "Unknown File";

        // Populate the text card
        authorDiv.textContent = "Author: " + d.label;
        fileDiv.textContent = "File: " + fileName;

        card.style.display = 'block';
        placeCard(ev.event.clientX, ev.event.clientY);
      }});

      gd.on('plotly_unhover', () => {{
        card.style.display = 'none';
      }});

      gd.addEventListener('mousemove', ev => {{
        if (card.style.display === 'block') placeCard(ev.clientX, ev.clientY);
      }});

    }}).catch(err => {{
      document.body.innerHTML = '<pre style="color:red">' + err + '</pre>';
    }});
  </script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(
        description="Generate a static t-SNE/UMAP HTML with hover text preview (no server needed)."
    )
    parser.add_argument("json_path", help="Path to umap_data.json")
    parser.add_argument("html_out", help="Output HTML file, e.g., index_hover.html")
    parser.add_argument("--title", default="Visualization",
                        help="Page and plot title (default: %(default)s)")
    parser.add_argument("--max-thumb", type=int, default=240,
                        help="Legacy parameter (ignored in text-only version)")
    parser.add_argument("--download-name", default="newplot", help="Filename for the downloaded PNG")
    args = parser.parse_args()

    json_path = Path(args.json_path)
    html_out = Path(args.html_out)

    # Load JSON, then minify for embedding
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    json_min = json.dumps(data, ensure_ascii=False, separators=(",", ":"))

    html = HTML_TEMPLATE.format(
        title=args.title,
        json_data=json_min,
        max_px=args.max_thumb,
        download_name=args.download_name
    )
    html_out.write_text(html, encoding="utf-8")
    print(f"✓ Wrote {html_out} — open it directly in your browser.")


if __name__ == "__main__":
    main()