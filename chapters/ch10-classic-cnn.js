window.CHAPTERS = window.CHAPTERS || [];
window.CHAPTERS.push({
  id: 'ch10',
  number: 10,
  title: 'Classic CNN Architectures',
  subtitle: 'From LeNet-5 to GoogLeNet: the landmark convolutional neural networks that shaped modern deep learning',
  sections: [

    // ═══════════════════════════════════════════════════════════════════════════
    // SECTION 1: LeNet-5
    // ═══════════════════════════════════════════════════════════════════════════
    {
      id: 'ch10-sec01',
      title: '1. LeNet-5',
      content: `
<div class="env-block intuition">
<strong>From Convolutions to Architectures.</strong>
Chapter 9 developed the convolution operation, pooling, and the idea of parameter sharing. This chapter puts those pieces together into complete architectures. We begin with the network that started it all: Yann LeCun's LeNet-5 (1998), the first CNN to demonstrate practical commercial success, reading handwritten digits on millions of checks for the US Postal Service.
</div>

<h2>LeNet-5: The Pioneer (LeCun et al., 1998)</h2>

<p>
LeNet-5 was designed for \\(32 \\times 32\\) grayscale images of handwritten digits (0 through 9). The architecture follows a pattern that every subsequent CNN would inherit: alternating convolution layers and subsampling (pooling) layers, followed by fully connected layers for classification.
</p>

<h3>Architecture Overview</h3>

<p>
The full architecture consists of seven learnable layers (not counting the input):
</p>

<ol>
  <li><strong>C1 (Convolution):</strong> 6 filters of size \\(5 \\times 5\\), producing 6 feature maps of size \\(28 \\times 28\\). Parameters: \\(6 \\times (5 \\times 5 \\times 1 + 1) = 156\\).</li>
  <li><strong>S2 (Subsampling):</strong> \\(2 \\times 2\\) average pooling with stride 2, reducing to 6 maps of size \\(14 \\times 14\\). Each pooling unit has a trainable weight and bias, yielding \\(6 \\times 2 = 12\\) parameters.</li>
  <li><strong>C3 (Convolution):</strong> 16 filters of size \\(5 \\times 5\\), producing 16 maps of \\(10 \\times 10\\). Not all input maps connect to all output maps; a specific connection table reduces parameters to \\(1{,}516\\).</li>
  <li><strong>S4 (Subsampling):</strong> \\(2 \\times 2\\) average pooling, reducing to 16 maps of \\(5 \\times 5\\). Parameters: \\(16 \\times 2 = 32\\).</li>
  <li><strong>C5 (Convolution):</strong> 120 filters of size \\(5 \\times 5\\), producing 120 maps of \\(1 \\times 1\\). This is effectively a fully connected layer. Parameters: \\(120 \\times (16 \\times 25 + 1) = 48{,}120\\).</li>
  <li><strong>F6 (Fully Connected):</strong> 84 units. Parameters: \\(84 \\times (120 + 1) = 10{,}164\\).</li>
  <li><strong>Output:</strong> 10 Euclidean Radial Basis Function units (one per digit class).</li>
</ol>

<p>
The total parameter count is approximately <strong>60,000</strong>, tiny by modern standards but revolutionary in 1998.
</p>

<div class="viz-placeholder" data-viz="viz-lenet5-arch"></div>

<h3>Key Design Decisions</h3>

<div class="env-block definition">
<strong>Activation Function.</strong>
LeNet-5 used the <em>scaled hyperbolic tangent</em>: \\(f(x) = 1.7159 \\tanh(\\tfrac{2}{3} x)\\). The constants were chosen so that (i) the effective range covers \\([-1, 1]\\), (ii) the second derivative is maximal at \\(x = 1\\), and (iii) the slope at the origin is close to 1 to avoid gradient attenuation in early layers.
</div>

<div class="env-block remark">
<strong>Sparse Connectivity in C3.</strong>
The C3 layer does not connect every one of the 6 input maps to every one of the 16 output maps. LeCun used a hand-designed connection table where each output map sees only a subset (3 or 4) of the input maps. The motivation was twofold: (i) keep the parameter count manageable, and (ii) force different maps to extract complementary features by breaking symmetry.
</div>

<h3>Why LeNet-5 Matters</h3>

<p>
LeNet-5 established the architectural template that all subsequent CNNs would follow:
</p>
<ul>
  <li><strong>Feature hierarchy:</strong> early layers detect edges, later layers detect parts, final layers recognize objects.</li>
  <li><strong>Spatial shrinkage:</strong> each conv+pool pair reduces spatial dimensions while increasing the number of channels.</li>
  <li><strong>End-to-end learning:</strong> every parameter, including the feature extractor and classifier, is trained jointly by backpropagation.</li>
</ul>

<div class="env-block warning">
<strong>Historical Context.</strong>
Despite its success on digit recognition, CNNs fell out of favor in the 2000s. The machine learning community pivoted to SVMs and hand-engineered features. It would take 14 years, until AlexNet in 2012, for deep CNNs to make their triumphant return, powered by GPUs and large datasets.
</div>
`,
      visualizations: [
        {
          id: 'viz-lenet5-arch',
          title: 'LeNet-5 Architecture Diagram',
          description: 'Hover over each layer to see dimensions and parameter counts. Click a layer to highlight it.',
          setup(container, controls) {
            const W = 820, H = 370;
            const viz = new VizEngine(container, { width: W, height: H, scale: 1, originX: 0, originY: 0 });
            const ctx = viz.ctx;

            const layers = [
              { name: 'Input',  type: 'input', dims: '32x32x1',    params: 0,      x: 30,  w: 40, h: 120, color: '#8b949e' },
              { name: 'C1',     type: 'conv',  dims: '28x28x6',    params: 156,    x: 110, w: 45, h: 105, color: '#58a6ff' },
              { name: 'S2',     type: 'pool',  dims: '14x14x6',    params: 12,     x: 195, w: 40, h: 70,  color: '#3fb9a0' },
              { name: 'C3',     type: 'conv',  dims: '10x10x16',   params: 1516,   x: 280, w: 50, h: 60,  color: '#58a6ff' },
              { name: 'S4',     type: 'pool',  dims: '5x5x16',     params: 32,     x: 370, w: 45, h: 40,  color: '#3fb9a0' },
              { name: 'C5',     type: 'conv',  dims: '1x1x120',    params: 48120,  x: 455, w: 55, h: 30,  color: '#58a6ff' },
              { name: 'F6',     type: 'fc',    dims: '84',         params: 10164,  x: 555, w: 50, h: 25,  color: '#bc8cff' },
              { name: 'Output', type: 'fc',    dims: '10',         params: 850,    x: 650, w: 45, h: 18,  color: '#f0883e' }
            ];

            let hoveredIdx = -1;
            let selectedIdx = -1;

            function typeLabel(t) {
              if (t === 'conv') return 'Convolution';
              if (t === 'pool') return 'Avg Pooling';
              if (t === 'fc')   return 'Fully Connected';
              return 'Input';
            }

            function draw() {
              ctx.fillStyle = viz.colors.bg;
              ctx.fillRect(0, 0, W, H);

              // Title
              ctx.fillStyle = viz.colors.white;
              ctx.font = 'bold 15px -apple-system,sans-serif';
              ctx.textAlign = 'center';
              ctx.textBaseline = 'top';
              ctx.fillText('LeNet-5 (LeCun et al., 1998)', W / 2, 10);

              const baseY = 170;

              // Draw connections
              for (let i = 0; i < layers.length - 1; i++) {
                const a = layers[i], b = layers[i + 1];
                ctx.strokeStyle = '#2a2a50';
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(a.x + a.w, baseY);
                ctx.lineTo(b.x, baseY);
                ctx.stroke();
              }

              // Draw layers
              layers.forEach((layer, i) => {
                const ly = baseY - layer.h / 2;
                const isHovered = i === hoveredIdx;
                const isSelected = i === selectedIdx;

                // Shadow / glow
                if (isHovered || isSelected) {
                  ctx.shadowColor = layer.color;
                  ctx.shadowBlur = 12;
                }

                // 3D effect: draw multiple offset rectangles for depth
                const depth = Math.max(3, Math.floor(layer.h / 15));
                for (let d = depth; d >= 0; d--) {
                  const alpha = d === 0 ? (isHovered ? 'cc' : 'aa') : '33';
                  ctx.fillStyle = layer.color + alpha;
                  ctx.strokeStyle = layer.color;
                  ctx.lineWidth = d === 0 ? 1.5 : 0.5;
                  const rx = layer.x + d * 2;
                  const ry = ly - d * 2;
                  ctx.fillRect(rx, ry, layer.w, layer.h);
                  ctx.strokeRect(rx, ry, layer.w, layer.h);
                }

                ctx.shadowColor = 'transparent';
                ctx.shadowBlur = 0;

                // Layer name
                ctx.fillStyle = viz.colors.white;
                ctx.font = 'bold 11px -apple-system,sans-serif';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'top';
                ctx.fillText(layer.name, layer.x + layer.w / 2, baseY + layer.h / 2 + 8);

                // Dimensions below
                ctx.fillStyle = viz.colors.text;
                ctx.font = '10px -apple-system,sans-serif';
                ctx.fillText(layer.dims, layer.x + layer.w / 2, baseY + layer.h / 2 + 22);
              });

              // Info panel for hovered/selected layer
              const infoIdx = hoveredIdx >= 0 ? hoveredIdx : selectedIdx;
              if (infoIdx >= 0) {
                const layer = layers[infoIdx];
                const px = 530, py = 280;
                ctx.fillStyle = '#14142edd';
                ctx.strokeStyle = layer.color;
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.roundRect(px, py, 270, 70, 8);
                ctx.fill();
                ctx.stroke();

                ctx.fillStyle = layer.color;
                ctx.font = 'bold 13px -apple-system,sans-serif';
                ctx.textAlign = 'left';
                ctx.textBaseline = 'top';
                ctx.fillText(layer.name + ' — ' + typeLabel(layer.type), px + 12, py + 10);

                ctx.fillStyle = viz.colors.text;
                ctx.font = '11px -apple-system,sans-serif';
                ctx.fillText('Output: ' + layer.dims, px + 12, py + 30);
                ctx.fillText('Parameters: ' + layer.params.toLocaleString(), px + 12, py + 46);
              }

              // Legend
              const lx = 30, ly2 = 290;
              const legendItems = [
                { color: '#58a6ff', label: 'Convolution' },
                { color: '#3fb9a0', label: 'Avg Pooling' },
                { color: '#bc8cff', label: 'Fully Connected' },
                { color: '#f0883e', label: 'Output' }
              ];
              legendItems.forEach((item, i) => {
                ctx.fillStyle = item.color + 'aa';
                ctx.fillRect(lx, ly2 + i * 18, 12, 12);
                ctx.fillStyle = viz.colors.text;
                ctx.font = '11px -apple-system,sans-serif';
                ctx.textAlign = 'left';
                ctx.textBaseline = 'top';
                ctx.fillText(item.label, lx + 18, ly2 + i * 18);
              });

              // Total params
              ctx.fillStyle = viz.colors.white;
              ctx.font = '11px -apple-system,sans-serif';
              ctx.textAlign = 'left';
              ctx.fillText('Total: ~60,000 parameters', lx, ly2 + legendItems.length * 18 + 6);
            }

            draw();

            // Mouse interaction
            viz.canvas.addEventListener('mousemove', (e) => {
              const rect = viz.canvas.getBoundingClientRect();
              const mx = (e.clientX - rect.left) * (W / rect.width);
              const my = (e.clientY - rect.top) * (H / rect.height);
              const baseY = 170;
              let found = -1;
              layers.forEach((layer, i) => {
                const ly = baseY - layer.h / 2;
                if (mx >= layer.x && mx <= layer.x + layer.w + 10 &&
                    my >= ly - 10 && my <= ly + layer.h + 10) {
                  found = i;
                }
              });
              if (found !== hoveredIdx) {
                hoveredIdx = found;
                viz.canvas.style.cursor = found >= 0 ? 'pointer' : 'default';
                draw();
              }
            });

            viz.canvas.addEventListener('click', (e) => {
              if (hoveredIdx >= 0) {
                selectedIdx = hoveredIdx === selectedIdx ? -1 : hoveredIdx;
                draw();
              }
            });

            viz.canvas.addEventListener('mouseleave', () => {
              hoveredIdx = -1;
              draw();
            });

            return viz;
          }
        }
      ],
      exercises: [
        {
          question: 'LeNet-5 processes \\(32 \\times 32\\) input images. After the C1 layer (6 filters of size \\(5 \\times 5\\), stride 1, no padding), what are the output dimensions? Verify using the convolution output formula.',
          hint: 'The formula is \\(\\lfloor (n - k) / s \\rfloor + 1\\), where \\(n\\) is the input size, \\(k\\) the kernel size, and \\(s\\) the stride.',
          solution: 'Using \\(\\lfloor (32 - 5)/1 \\rfloor + 1 = 28\\), the output is \\(28 \\times 28\\). With 6 filters, the full output shape is \\(28 \\times 28 \\times 6\\).'
        },
        {
          question: 'In the C1 layer, each of the 6 filters has size \\(5 \\times 5\\) and operates on a single input channel. Count the total number of learnable parameters, including biases.',
          hint: 'Each filter has \\(5 \\times 5 = 25\\) weights plus 1 bias.',
          solution: 'Each filter: \\(5 \\times 5 + 1 = 26\\) parameters. With 6 filters: \\(6 \\times 26 = 156\\) parameters.'
        },
        {
          question: 'Why did LeCun use a sparse connection table in the C3 layer instead of connecting all 6 input feature maps to all 16 output maps? Give two reasons.',
          hint: 'Think about both computational cost and feature diversity.',
          solution: 'Two reasons: (1) <strong>Reduced parameters:</strong> full connectivity would require \\(16 \\times 6 \\times 25 = 2{,}400\\) weights per output map; sparse connectivity reduces this. (2) <strong>Symmetry breaking:</strong> by forcing different output maps to see different subsets of input maps, the network learns complementary features rather than redundant ones.'
        }
      ]
    },

    // ═══════════════════════════════════════════════════════════════════════════
    // SECTION 2: AlexNet
    // ═══════════════════════════════════════════════════════════════════════════
    {
      id: 'ch10-sec02',
      title: '2. AlexNet',
      content: `
<h2>AlexNet: The Deep Learning Revolution (Krizhevsky et al., 2012)</h2>

<div class="env-block intuition">
<strong>The ImageNet Moment.</strong>
In 2012, Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton submitted a CNN to the ImageNet Large Scale Visual Recognition Challenge (ILSVRC). Their network, AlexNet, achieved a top-5 error rate of 15.3%, crushing the second-place entry (26.2%) which used hand-engineered features. This result shattered the prevailing wisdom that deep learning could not scale to real-world vision tasks, and triggered the deep learning revolution that continues today.
</div>

<h3>Architecture</h3>

<p>
AlexNet processes \\(224 \\times 224 \\times 3\\) (RGB) images through 5 convolutional layers and 3 fully connected layers. The network was split across two GPUs due to memory limitations of 2012-era hardware (GTX 580, 3GB VRAM each).
</p>

<div class="env-block definition">
<strong>AlexNet Layer-by-Layer.</strong>
<ol>
  <li><strong>Conv1:</strong> 96 filters of \\(11 \\times 11\\), stride 4 \\(\\to\\) \\(55 \\times 55 \\times 96\\). Params: 34,944.</li>
  <li><strong>MaxPool1:</strong> \\(3 \\times 3\\), stride 2 \\(\\to\\) \\(27 \\times 27 \\times 96\\).</li>
  <li><strong>Conv2:</strong> 256 filters of \\(5 \\times 5\\), pad 2 \\(\\to\\) \\(27 \\times 27 \\times 256\\). Params: 614,656.</li>
  <li><strong>MaxPool2:</strong> \\(3 \\times 3\\), stride 2 \\(\\to\\) \\(13 \\times 13 \\times 256\\).</li>
  <li><strong>Conv3:</strong> 384 filters of \\(3 \\times 3\\), pad 1 \\(\\to\\) \\(13 \\times 13 \\times 384\\). Params: 885,120.</li>
  <li><strong>Conv4:</strong> 384 filters of \\(3 \\times 3\\), pad 1 \\(\\to\\) \\(13 \\times 13 \\times 384\\). Params: 1,327,488.</li>
  <li><strong>Conv5:</strong> 256 filters of \\(3 \\times 3\\), pad 1 \\(\\to\\) \\(13 \\times 13 \\times 256\\). Params: 884,992.</li>
  <li><strong>MaxPool3:</strong> \\(3 \\times 3\\), stride 2 \\(\\to\\) \\(6 \\times 6 \\times 256\\).</li>
  <li><strong>FC6:</strong> 4,096 units. Params: 37,752,832.</li>
  <li><strong>FC7:</strong> 4,096 units. Params: 16,781,312.</li>
  <li><strong>FC8 (Output):</strong> 1,000 units (ImageNet classes). Params: 4,097,000.</li>
</ol>
Total: approximately <strong>62.4 million</strong> parameters. Note that the FC layers alone account for over 94% of all parameters.
</div>

<div class="viz-placeholder" data-viz="viz-alexnet-arch"></div>

<h3>Key Innovations</h3>

<h4>1. ReLU Activation</h4>
<p>
AlexNet replaced the traditional sigmoid/tanh with the <em>Rectified Linear Unit</em>:
</p>
\\[
\\text{ReLU}(x) = \\max(0, x)
\\]
<p>
ReLU trains roughly 6x faster than tanh (as reported in the original paper) because it avoids the saturation problem: for large positive inputs, the gradient is exactly 1 rather than exponentially small. This was one of the most influential design choices in the history of deep learning.
</p>

<h4>2. Dropout Regularization</h4>
<p>
The fully connected layers use <em>dropout</em> with probability \\(p = 0.5\\). During training, each neuron's output is randomly set to zero with probability \\(p\\). At test time, all neurons are active but their outputs are multiplied by \\((1-p)\\). Dropout acts as an implicit ensemble of \\(2^n\\) subnetworks and was essential to prevent overfitting in the massive FC layers.
</p>

<h4>3. Data Augmentation</h4>
<p>
AlexNet applied two forms of data augmentation at training time:
</p>
<ul>
  <li><strong>Random crops and flips:</strong> from each \\(256 \\times 256\\) image, random \\(224 \\times 224\\) patches were extracted (with horizontal flips), increasing the effective dataset by a factor of 2,048.</li>
  <li><strong>PCA color jittering:</strong> adding multiples of the principal components of the RGB pixel values to each training image, making the network more robust to illumination changes.</li>
</ul>

<h4>4. Local Response Normalization (LRN)</h4>
<p>
AlexNet introduced <em>local response normalization</em> across channels:
</p>
\\[
b_{x,y}^i = \\frac{a_{x,y}^i}{\\left(k + \\alpha \\sum_{j=\\max(0,i-n/2)}^{\\min(N-1,i+n/2)} (a_{x,y}^j)^2 \\right)^\\beta}
\\]
<p>
This implements a form of lateral inhibition inspired by neuroscience. However, LRN was later shown to provide minimal benefit and was replaced by batch normalization in subsequent architectures.
</p>

<div class="env-block remark">
<strong>GPU Training.</strong>
AlexNet was one of the first networks trained on GPUs. The model was split across two NVIDIA GTX 580 GPUs. Certain layers communicated between GPUs while others operated independently. Training took 5 to 6 days. This GPU training paradigm became standard and accelerated progress enormously.
</div>
`,
      visualizations: [
        {
          id: 'viz-alexnet-arch',
          title: 'AlexNet Architecture with Parameter Counts',
          description: 'Each bar shows parameter count per layer. The fully connected layers dominate. Hover for details.',
          setup(container, controls) {
            const W = 820, H = 420;
            const viz = new VizEngine(container, { width: W, height: H, scale: 1, originX: 0, originY: 0 });
            const ctx = viz.ctx;

            const layers = [
              { name: 'Conv1',    dims: '55x55x96',   params: 34944,     type: 'conv',  ksize: '11x11, s4' },
              { name: 'Pool1',    dims: '27x27x96',   params: 0,         type: 'pool',  ksize: '3x3, s2' },
              { name: 'Conv2',    dims: '27x27x256',  params: 614656,    type: 'conv',  ksize: '5x5, p2' },
              { name: 'Pool2',    dims: '13x13x256',  params: 0,         type: 'pool',  ksize: '3x3, s2' },
              { name: 'Conv3',    dims: '13x13x384',  params: 885120,    type: 'conv',  ksize: '3x3, p1' },
              { name: 'Conv4',    dims: '13x13x384',  params: 1327488,   type: 'conv',  ksize: '3x3, p1' },
              { name: 'Conv5',    dims: '13x13x256',  params: 884992,    type: 'conv',  ksize: '3x3, p1' },
              { name: 'Pool3',    dims: '6x6x256',    params: 0,         type: 'pool',  ksize: '3x3, s2' },
              { name: 'FC6',      dims: '4096',        params: 37752832,  type: 'fc',    ksize: '+Dropout' },
              { name: 'FC7',      dims: '4096',        params: 16781312,  type: 'fc',    ksize: '+Dropout' },
              { name: 'FC8',      dims: '1000',        params: 4097000,   type: 'fc',    ksize: 'Softmax' }
            ];

            const maxParams = 37752832;
            const barAreaTop = 60;
            const barAreaBottom = 280;
            const barAreaH = barAreaBottom - barAreaTop;
            const barW = 52;
            const gap = 16;
            const startX = 40;

            let hoveredIdx = -1;

            function getColor(type) {
              if (type === 'conv') return '#58a6ff';
              if (type === 'pool') return '#3fb9a0';
              if (type === 'fc')   return '#bc8cff';
              return '#8b949e';
            }

            function draw() {
              ctx.fillStyle = viz.colors.bg;
              ctx.fillRect(0, 0, W, H);

              ctx.fillStyle = viz.colors.white;
              ctx.font = 'bold 15px -apple-system,sans-serif';
              ctx.textAlign = 'center';
              ctx.textBaseline = 'top';
              ctx.fillText('AlexNet — Parameter Distribution by Layer', W / 2, 12);

              ctx.fillStyle = viz.colors.text;
              ctx.font = '11px -apple-system,sans-serif';
              ctx.fillText('62.4M total parameters  |  Input: 224x224x3  |  Output: 1000 classes', W / 2, 34);

              // Draw bars
              layers.forEach((layer, i) => {
                const x = startX + i * (barW + gap);
                const logMax = Math.log10(maxParams + 1);
                const barH = layer.params > 0
                  ? Math.max(6, (Math.log10(layer.params + 1) / logMax) * barAreaH)
                  : 4;
                const y = barAreaBottom - barH;
                const color = getColor(layer.type);
                const isHovered = i === hoveredIdx;

                // Bar
                ctx.fillStyle = isHovered ? color + 'ee' : color + '99';
                ctx.fillRect(x, y, barW, barH);
                ctx.strokeStyle = color;
                ctx.lineWidth = isHovered ? 2 : 1;
                ctx.strokeRect(x, y, barW, barH);

                // Layer name (rotated)
                ctx.save();
                ctx.translate(x + barW / 2, barAreaBottom + 8);
                ctx.fillStyle = viz.colors.text;
                ctx.font = '10px -apple-system,sans-serif';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'top';
                ctx.fillText(layer.name, 0, 0);
                ctx.restore();

                // Param count on top of bar
                if (layer.params > 0) {
                  ctx.fillStyle = viz.colors.white;
                  ctx.font = '9px -apple-system,sans-serif';
                  ctx.textAlign = 'center';
                  ctx.textBaseline = 'bottom';
                  const pStr = layer.params >= 1000000
                    ? (layer.params / 1000000).toFixed(1) + 'M'
                    : layer.params >= 1000
                      ? (layer.params / 1000).toFixed(0) + 'K'
                      : layer.params.toString();
                  ctx.fillText(pStr, x + barW / 2, y - 3);
                }
              });

              // Y-axis label
              ctx.save();
              ctx.translate(14, (barAreaTop + barAreaBottom) / 2);
              ctx.rotate(-Math.PI / 2);
              ctx.fillStyle = viz.colors.text;
              ctx.font = '10px -apple-system,sans-serif';
              ctx.textAlign = 'center';
              ctx.fillText('Parameters (log scale)', 0, 0);
              ctx.restore();

              // Dims label row
              layers.forEach((layer, i) => {
                const x = startX + i * (barW + gap);
                ctx.fillStyle = '#6e7681';
                ctx.font = '9px -apple-system,sans-serif';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'top';
                ctx.fillText(layer.dims, x + barW / 2, barAreaBottom + 22);
              });

              // Info panel on hover
              if (hoveredIdx >= 0) {
                const layer = layers[hoveredIdx];
                const px = 500, py = 320;
                const color = getColor(layer.type);
                ctx.fillStyle = '#14142edd';
                ctx.strokeStyle = color;
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.roundRect(px, py, 300, 85, 8);
                ctx.fill();
                ctx.stroke();

                ctx.fillStyle = color;
                ctx.font = 'bold 13px -apple-system,sans-serif';
                ctx.textAlign = 'left';
                ctx.textBaseline = 'top';
                ctx.fillText(layer.name + ' (' + layer.type.toUpperCase() + ')', px + 12, py + 10);

                ctx.fillStyle = viz.colors.text;
                ctx.font = '11px -apple-system,sans-serif';
                ctx.fillText('Output shape: ' + layer.dims, px + 12, py + 30);
                ctx.fillText('Config: ' + layer.ksize, px + 12, py + 46);
                ctx.fillText('Parameters: ' + layer.params.toLocaleString(), px + 12, py + 62);
              }

              // Legend
              const lx = 30, ly = 320;
              [
                { c: '#58a6ff', l: 'Convolution' },
                { c: '#3fb9a0', l: 'Max Pooling' },
                { c: '#bc8cff', l: 'Fully Connected' }
              ].forEach((item, i) => {
                ctx.fillStyle = item.c + 'aa';
                ctx.fillRect(lx, ly + i * 18, 12, 12);
                ctx.fillStyle = viz.colors.text;
                ctx.font = '11px -apple-system,sans-serif';
                ctx.textAlign = 'left';
                ctx.textBaseline = 'top';
                ctx.fillText(item.l, lx + 18, ly + i * 18);
              });

              // Percentage annotation
              ctx.fillStyle = '#f0883e';
              ctx.font = '11px -apple-system,sans-serif';
              ctx.textAlign = 'left';
              ctx.fillText('FC layers: 94% of all parameters', lx, ly + 3 * 18 + 6);
            }

            draw();

            viz.canvas.addEventListener('mousemove', (e) => {
              const rect = viz.canvas.getBoundingClientRect();
              const mx = (e.clientX - rect.left) * (W / rect.width);
              let found = -1;
              layers.forEach((_, i) => {
                const x = startX + i * (barW + gap);
                if (mx >= x && mx <= x + barW) found = i;
              });
              if (found !== hoveredIdx) {
                hoveredIdx = found;
                viz.canvas.style.cursor = found >= 0 ? 'pointer' : 'default';
                draw();
              }
            });

            viz.canvas.addEventListener('mouseleave', () => {
              hoveredIdx = -1;
              draw();
            });

            return viz;
          }
        }
      ],
      exercises: [
        {
          question: 'Compute the output spatial dimensions of Conv1 in AlexNet. The input is \\(224 \\times 224\\), the kernel is \\(11 \\times 11\\), stride is 4, and padding is 0 (some implementations use padding 2 with input 227).',
          hint: 'Use the formula \\(\\lfloor (n + 2p - k) / s \\rfloor + 1\\).',
          solution: 'Without padding: \\(\\lfloor (224 - 11)/4 \\rfloor + 1 = \\lfloor 213/4 \\rfloor + 1 = 53 + 1 = 54\\). The original paper reports \\(55 \\times 55\\), which corresponds to using \\(227 \\times 227\\) input: \\(\\lfloor (227 - 11)/4 \\rfloor + 1 = 55\\). (Different implementations use slightly different input sizes; the key insight is the same.)'
        },
        {
          question: 'Why do the fully connected layers in AlexNet contain the vast majority (over 94%) of the parameters? What architectural insight does this suggest for future designs?',
          hint: 'Consider the dimensionality at the FC6 input: the flattened feature map from Pool3.',
          solution: 'FC6 takes a flattened \\(6 \\times 6 \\times 256 = 9{,}216\\)-dimensional input and maps it to 4,096 units, requiring \\(9{,}216 \\times 4{,}096 \\approx 37.7\\text{M}\\) weights alone. Convolutional layers share parameters across spatial locations, making them parameter-efficient. This disparity suggests that reducing or eliminating FC layers (as VGG, GoogLeNet, and ResNet later did with global average pooling) can dramatically shrink model size without hurting accuracy.'
        },
        {
          question: 'Explain how dropout with \\(p = 0.5\\) acts as an implicit model ensemble. How many distinct subnetworks does a single dropout layer over \\(n\\) neurons effectively average over?',
          hint: 'Each neuron is independently either active or dropped. How many binary configurations exist for \\(n\\) neurons?',
          solution: 'Each of the \\(n\\) neurons is independently kept (probability 0.5) or dropped (probability 0.5), giving \\(2^n\\) possible subnetworks. For FC6 with \\(n = 4{,}096\\), this is \\(2^{4096}\\), an astronomically large ensemble. At test time, using all neurons with weights scaled by \\(1-p\\) approximates the geometric mean of predictions across all subnetworks.'
        }
      ]
    },

    // ═══════════════════════════════════════════════════════════════════════════
    // SECTION 3: VGGNet
    // ═══════════════════════════════════════════════════════════════════════════
    {
      id: 'ch10-sec03',
      title: '3. VGGNet',
      content: `
<h2>VGGNet: The Elegance of Depth (Simonyan & Zisserman, 2014)</h2>

<div class="env-block intuition">
<strong>One Kernel Size to Rule Them All.</strong>
Where AlexNet used a grab-bag of kernel sizes (11, 5, 3), the Visual Geometry Group at Oxford asked a radical question: what if we use <em>only</em> \\(3 \\times 3\\) convolutions and simply stack more of them? The answer, VGGNet, demonstrated that depth, rather than kernel size, is the critical factor for representation power. VGG-16 and VGG-19 achieved second place at ILSVRC 2014 and became the default feature extractor for transfer learning.
</div>

<h3>The Core Insight: Stacked 3x3 Convolutions</h3>

<p>
The fundamental observation behind VGGNet is that stacking small convolutions achieves the same <em>effective receptive field</em> as a single large convolution, but with fewer parameters and more non-linearities.
</p>

<div class="env-block theorem">
<strong>Proposition (Receptive Field of Stacked Convolutions).</strong>
A stack of \\(n\\) convolutional layers, each with kernel size \\(3 \\times 3\\) and stride 1, has an effective receptive field of \\((2n + 1) \\times (2n + 1)\\). Thus:
<ul>
  <li>2 layers of \\(3 \\times 3\\) \\(\\equiv\\) one layer of \\(5 \\times 5\\) (receptive field),</li>
  <li>3 layers of \\(3 \\times 3\\) \\(\\equiv\\) one layer of \\(7 \\times 7\\) (receptive field).</li>
</ul>
</div>

<div class="env-block proof">
<strong>Proof.</strong>
Consider a single \\(3 \\times 3\\) convolution: each output pixel depends on a \\(3 \\times 3\\) patch of the input. A second \\(3 \\times 3\\) convolution applied to this output produces pixels that each depend on a \\(3 \\times 3\\) patch of the first output, which itself depends on input. The union of those input dependencies forms a \\(5 \\times 5\\) patch. By induction, after \\(n\\) layers the receptive field is \\((2n+1) \\times (2n+1)\\). <span class="qed">\\(\\square\\)</span>
</div>

<h3>Parameter Efficiency</h3>

<p>
A single \\(7 \\times 7\\) convolution with \\(C\\) input and \\(C\\) output channels has \\(7^2 C^2 = 49C^2\\) parameters. Three stacked \\(3 \\times 3\\) convolutions have \\(3 \\times 3^2 C^2 = 27C^2\\) parameters, a <strong>45% reduction</strong>. Additionally, the three-layer stack inserts two extra ReLU non-linearities, making the decision function more discriminative.
</p>

<h3>VGG-16 Architecture</h3>

<p>
VGG-16 consists of 13 convolutional layers and 3 fully connected layers, organized in 5 "blocks":
</p>

<table style="width:100%;border-collapse:collapse;margin:1rem 0;">
  <tr style="background:#1a1a40;">
    <th style="padding:8px;border:1px solid #30363d;">Block</th>
    <th style="padding:8px;border:1px solid #30363d;">Layers</th>
    <th style="padding:8px;border:1px solid #30363d;">Output Size</th>
    <th style="padding:8px;border:1px solid #30363d;">Channels</th>
  </tr>
  <tr>
    <td style="padding:8px;border:1px solid #30363d;">Block 1</td>
    <td style="padding:8px;border:1px solid #30363d;">2 x Conv3-64 + MaxPool</td>
    <td style="padding:8px;border:1px solid #30363d;">112x112</td>
    <td style="padding:8px;border:1px solid #30363d;">64</td>
  </tr>
  <tr>
    <td style="padding:8px;border:1px solid #30363d;">Block 2</td>
    <td style="padding:8px;border:1px solid #30363d;">2 x Conv3-128 + MaxPool</td>
    <td style="padding:8px;border:1px solid #30363d;">56x56</td>
    <td style="padding:8px;border:1px solid #30363d;">128</td>
  </tr>
  <tr>
    <td style="padding:8px;border:1px solid #30363d;">Block 3</td>
    <td style="padding:8px;border:1px solid #30363d;">3 x Conv3-256 + MaxPool</td>
    <td style="padding:8px;border:1px solid #30363d;">28x28</td>
    <td style="padding:8px;border:1px solid #30363d;">256</td>
  </tr>
  <tr>
    <td style="padding:8px;border:1px solid #30363d;">Block 4</td>
    <td style="padding:8px;border:1px solid #30363d;">3 x Conv3-512 + MaxPool</td>
    <td style="padding:8px;border:1px solid #30363d;">14x14</td>
    <td style="padding:8px;border:1px solid #30363d;">512</td>
  </tr>
  <tr>
    <td style="padding:8px;border:1px solid #30363d;">Block 5</td>
    <td style="padding:8px;border:1px solid #30363d;">3 x Conv3-512 + MaxPool</td>
    <td style="padding:8px;border:1px solid #30363d;">7x7</td>
    <td style="padding:8px;border:1px solid #30363d;">512</td>
  </tr>
  <tr>
    <td style="padding:8px;border:1px solid #30363d;">Classifier</td>
    <td style="padding:8px;border:1px solid #30363d;">FC-4096, FC-4096, FC-1000</td>
    <td style="padding:8px;border:1px solid #30363d;">--</td>
    <td style="padding:8px;border:1px solid #30363d;">--</td>
  </tr>
</table>

<p>
Total parameters: approximately <strong>138 million</strong> (VGG-16) or <strong>144 million</strong> (VGG-19, which adds one extra conv layer to blocks 3, 4, and 5). The model's beauty lies in its homogeneity: every convolution is \\(3 \\times 3\\) with stride 1 and padding 1; every pooling is \\(2 \\times 2\\) with stride 2.
</p>

<div class="viz-placeholder" data-viz="viz-vgg-receptive"></div>

<div class="env-block remark">
<strong>VGG's Legacy.</strong>
Although VGG's parameter count is prohibitively large (the model occupies over 500 MB), its convolutional features became the standard backbone for transfer learning in tasks such as object detection (Faster R-CNN), style transfer (Gatys et al., 2015), and perceptual loss functions. The principle of using only \\(3 \\times 3\\) convolutions became canonical and was adopted by nearly every architecture that followed.
</div>
`,
      visualizations: [
        {
          id: 'viz-vgg-receptive',
          title: 'VGG Receptive Field Growth',
          description: 'Use the slider to add stacked 3x3 convolution layers and watch the effective receptive field grow. Compare with a single large kernel.',
          setup(container, controls) {
            const W = 760, H = 420;
            const viz = new VizEngine(container, { width: W, height: H, scale: 1, originX: 0, originY: 0 });
            const ctx = viz.ctx;

            let numLayers = 1;

            function draw() {
              ctx.fillStyle = viz.colors.bg;
              ctx.fillRect(0, 0, W, H);

              ctx.fillStyle = viz.colors.white;
              ctx.font = 'bold 14px -apple-system,sans-serif';
              ctx.textAlign = 'center';
              ctx.textBaseline = 'top';
              ctx.fillText('Receptive Field Growth: Stacked 3x3 Convolutions', W / 2, 10);

              const n = numLayers;
              const rf = 2 * n + 1;
              const cellSize = Math.min(28, Math.floor(260 / rf));
              const gridStartX = 100;
              const gridStartY = 70;

              // Draw input grid (receptive field)
              ctx.fillStyle = viz.colors.text;
              ctx.font = '12px -apple-system,sans-serif';
              ctx.textAlign = 'center';
              ctx.fillText('Input Grid (Receptive Field = ' + rf + 'x' + rf + ')', gridStartX + rf * cellSize / 2, gridStartY - 18);

              for (let r = 0; r < rf; r++) {
                for (let c = 0; c < rf; c++) {
                  const x = gridStartX + c * cellSize;
                  const y = gridStartY + r * cellSize;

                  // Color based on which layer "sees" this cell
                  const dist = Math.max(Math.abs(r - n), Math.abs(c - n));
                  const layerIdx = n - dist;  // which layer first covers this cell

                  const colors = ['#58a6ff', '#3fb9a0', '#f0883e', '#bc8cff', '#f85149', '#d29922'];
                  const col = layerIdx >= 0 ? colors[layerIdx % colors.length] : '#1a1a40';
                  const alpha = layerIdx >= 0 ? '88' : '44';
                  ctx.fillStyle = col + alpha;
                  ctx.fillRect(x, y, cellSize - 1, cellSize - 1);
                  ctx.strokeStyle = col + 'aa';
                  ctx.lineWidth = 0.5;
                  ctx.strokeRect(x, y, cellSize - 1, cellSize - 1);
                }
              }

              // Center pixel highlight
              const cx = gridStartX + n * cellSize;
              const cy = gridStartY + n * cellSize;
              ctx.strokeStyle = viz.colors.white;
              ctx.lineWidth = 2;
              ctx.strokeRect(cx, cy, cellSize - 1, cellSize - 1);

              // Layer stack visualization on the right
              const stackX = 480;
              const stackY = 80;
              const layerH = 50;
              const layerW = 220;

              ctx.fillStyle = viz.colors.text;
              ctx.font = '12px -apple-system,sans-serif';
              ctx.textAlign = 'center';
              ctx.fillText('Layer Stack', stackX + layerW / 2, stackY - 18);

              // Output pixel
              ctx.fillStyle = viz.colors.white + '33';
              ctx.fillRect(stackX + layerW / 2 - 15, stackY - 5, 30, 20);
              ctx.strokeStyle = viz.colors.white;
              ctx.lineWidth = 1;
              ctx.strokeRect(stackX + layerW / 2 - 15, stackY - 5, 30, 20);
              ctx.fillStyle = viz.colors.white;
              ctx.font = '9px -apple-system,sans-serif';
              ctx.fillText('1x1 output', stackX + layerW / 2, stackY + 22);

              for (let i = 0; i < n; i++) {
                const ly = stackY + 40 + i * (layerH + 12);
                const rfAtLayer = 2 * (i + 1) + 1;
                const widthFrac = Math.min(1, rfAtLayer / rf);
                const bw = layerW * widthFrac;
                const bx = stackX + (layerW - bw) / 2;

                const colors = ['#58a6ff', '#3fb9a0', '#f0883e', '#bc8cff', '#f85149', '#d29922'];
                const col = colors[i % colors.length];

                ctx.fillStyle = col + '55';
                ctx.fillRect(bx, ly, bw, layerH - 6);
                ctx.strokeStyle = col;
                ctx.lineWidth = 1.5;
                ctx.strokeRect(bx, ly, bw, layerH - 6);

                ctx.fillStyle = viz.colors.white;
                ctx.font = '11px -apple-system,sans-serif';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText('Conv 3x3 (layer ' + (i + 1) + ')', stackX + layerW / 2, ly + (layerH - 6) / 2);

                ctx.fillStyle = viz.colors.text;
                ctx.font = '10px -apple-system,sans-serif';
                ctx.textBaseline = 'top';
                ctx.fillText('RF: ' + rfAtLayer + 'x' + rfAtLayer, stackX + layerW / 2, ly + layerH - 2);

                // Arrow between layers
                if (i < n - 1) {
                  ctx.strokeStyle = '#4a4a7a';
                  ctx.lineWidth = 1;
                  ctx.beginPath();
                  ctx.moveTo(stackX + layerW / 2, ly + layerH - 6);
                  ctx.lineTo(stackX + layerW / 2, ly + layerH + 6);
                  ctx.stroke();
                  ctx.beginPath();
                  ctx.moveTo(stackX + layerW / 2 - 4, ly + layerH + 2);
                  ctx.lineTo(stackX + layerW / 2, ly + layerH + 6);
                  ctx.lineTo(stackX + layerW / 2 + 4, ly + layerH + 2);
                  ctx.fill();
                }
              }

              // Comparison box
              const cmpY = 340;
              ctx.fillStyle = '#14142edd';
              ctx.strokeStyle = '#30363d';
              ctx.lineWidth = 1;
              ctx.beginPath();
              ctx.roundRect(30, cmpY, W - 60, 65, 8);
              ctx.fill();
              ctx.stroke();

              const singleParams = rf * rf;
              const stackedParams = n * 9;
              const saving = singleParams > 0 ? Math.round((1 - stackedParams / singleParams) * 100) : 0;

              ctx.fillStyle = viz.colors.white;
              ctx.font = 'bold 12px -apple-system,sans-serif';
              ctx.textAlign = 'left';
              ctx.textBaseline = 'top';
              ctx.fillText('Parameter Comparison (per channel, C=1):', 50, cmpY + 10);

              ctx.font = '11px -apple-system,sans-serif';
              ctx.fillStyle = '#f0883e';
              ctx.fillText('Single ' + rf + 'x' + rf + ' kernel: ' + singleParams + ' params', 50, cmpY + 30);
              ctx.fillStyle = '#58a6ff';
              ctx.fillText(n + ' stacked 3x3: ' + n + ' x 9 = ' + stackedParams + ' params', 350, cmpY + 30);
              ctx.fillStyle = n > 1 ? '#3fb950' : viz.colors.text;
              ctx.fillText(n > 1 ? 'Saving: ' + saving + '% fewer parameters + ' + (n - 1) + ' extra ReLU(s)' : 'Add more layers to see the savings', 50, cmpY + 47);
            }

            draw();

            VizEngine.createSlider(controls, 'Layers', 1, 5, 1, 1, (val) => {
              numLayers = Math.round(val);
              draw();
            });

            return viz;
          }
        }
      ],
      exercises: [
        {
          question: 'Verify the parameter savings. For a network with \\(C\\) input and \\(C\\) output channels, compare the parameter count of (a) a single \\(5 \\times 5\\) conv layer vs. (b) two stacked \\(3 \\times 3\\) conv layers. Express both in terms of \\(C\\).',
          hint: 'A conv layer with \\(k \\times k\\) kernel, \\(C_{in}\\) input channels, and \\(C_{out}\\) output channels has \\(k^2 C_{in} C_{out}\\) weight parameters (ignoring biases).',
          solution: '(a) \\(5^2 \\cdot C \\cdot C = 25C^2\\). (b) First layer: \\(3^2 \\cdot C \\cdot C = 9C^2\\); second layer: \\(3^2 \\cdot C \\cdot C = 9C^2\\); total: \\(18C^2\\). The stacked version saves \\(1 - 18/25 = 28\\%\\) of the parameters while matching the \\(5 \\times 5\\) receptive field.'
        },
        {
          question: 'In VGG-16, after Block 3 (three Conv3-256 layers + MaxPool), the spatial size is \\(28 \\times 28\\) with 256 channels. How many multiply-add operations does one Conv3-256 layer in this block perform? (Ignore biases.)',
          hint: 'Each output pixel requires \\(k^2 \\times C_{in}\\) multiplications. The output has \\(H \\times W \\times C_{out}\\) pixels.',
          solution: 'Output: \\(28 \\times 28 \\times 256\\). Each output pixel: \\(3^2 \\times 256 = 2{,}304\\) multiplications. Total: \\(28 \\times 28 \\times 256 \\times 2{,}304 = 462{,}422{,}016 \\approx 462\\text{M}\\) multiply-adds per layer.'
        },
        {
          question: 'Why is VGG-16 still widely used as a feature extractor for transfer learning despite its enormous size (138M parameters)?',
          hint: 'Consider the quality of learned features, the simplicity of the architecture, and what happens when you remove the FC layers.',
          solution: 'Three reasons: (1) VGG learns excellent hierarchical features because its depth and homogeneous design allow each block to specialize. (2) The convolutional blocks alone (without FC layers) contain only 14.7M parameters but produce rich features. (3) The architecture is extremely simple and predictable (just 3x3 convs + max pool), making it easy to modify, fine-tune, and understand. These features (especially after stripping the FC layers) make it ideal for tasks like style transfer, perceptual loss, and detection backbones.'
        }
      ]
    },

    // ═══════════════════════════════════════════════════════════════════════════
    // SECTION 4: GoogLeNet & Inception
    // ═══════════════════════════════════════════════════════════════════════════
    {
      id: 'ch10-sec04',
      title: '4. GoogLeNet & Inception',
      content: `
<h2>GoogLeNet & Inception: Multi-Scale Feature Extraction (Szegedy et al., 2015)</h2>

<div class="env-block intuition">
<strong>Going Deeper, Smarter.</strong>
While VGG went deeper with brute force (stacking identical blocks), the Google team asked a more subtle question: instead of choosing one filter size per layer, why not use <em>multiple sizes simultaneously</em> and let the network decide which scales matter? This led to the <em>Inception module</em>, which processes input through parallel branches of different kernel sizes and concatenates their outputs. GoogLeNet (22 layers deep) won ILSVRC 2014 with only <strong>6.8 million</strong> parameters, 20x fewer than VGG-16.
</div>

<h3>The Inception Module</h3>

<p>
An Inception module applies four parallel operations to the same input:
</p>

<ol>
  <li><strong>1x1 convolution</strong> branch: captures channel-wise (pointwise) interactions.</li>
  <li><strong>1x1 conv \\(\\to\\) 3x3 conv</strong> branch: reduces dimensions first, then captures local spatial patterns.</li>
  <li><strong>1x1 conv \\(\\to\\) 5x5 conv</strong> branch: reduces dimensions first, then captures broader spatial patterns.</li>
  <li><strong>3x3 max pool \\(\\to\\) 1x1 conv</strong> branch: preserves spatial information through pooling, then reduces channels.</li>
</ol>

<p>
The outputs of all four branches are concatenated along the channel axis, producing a single output tensor.
</p>

<div class="viz-placeholder" data-viz="viz-inception-module"></div>

<h3>The Role of 1x1 Convolutions</h3>

<div class="env-block definition">
<strong>Bottleneck Layer (1x1 Convolution).</strong>
A \\(1 \\times 1\\) convolution with \\(C_{out} &lt; C_{in}\\) channels computes a linear combination across channels at each spatial location, acting as a <em>dimensionality reduction</em> step. It reduces the number of input channels before applying the expensive \\(3 \\times 3\\) or \\(5 \\times 5\\) convolutions, dramatically cutting the computational cost.
</div>

<div class="env-block example">
<strong>Example: Computational savings.</strong>
Suppose the input is \\(28 \\times 28 \\times 256\\). A direct \\(5 \\times 5\\) convolution to produce 128 channels costs:
\\[
28 \\times 28 \\times 128 \\times (5^2 \\times 256) = 642\\text{M multiply-adds.}
\\]
With a 1x1 bottleneck reducing to 32 channels first:
\\[
\\underbrace{28 \\times 28 \\times 32 \\times 256}_{\\text{1x1 conv: 6.4M}} + \\underbrace{28 \\times 28 \\times 128 \\times (25 \\times 32)}_{\\text{5x5 conv: 80.3M}} = 86.7\\text{M,}
\\]
a <strong>7.4x reduction</strong> in computation.
</div>

<h3>GoogLeNet Architecture Overview</h3>

<p>
GoogLeNet stacks 9 Inception modules across 22 layers:
</p>

<ul>
  <li><strong>Stem:</strong> Conv7x7/2, MaxPool3x3/2, Conv1x1, Conv3x3, MaxPool3x3/2</li>
  <li><strong>Inception 3a, 3b:</strong> Two inception modules + MaxPool</li>
  <li><strong>Inception 4a, 4b, 4c, 4d, 4e:</strong> Five inception modules + MaxPool</li>
  <li><strong>Inception 5a, 5b:</strong> Two inception modules</li>
  <li><strong>Classifier:</strong> Global Average Pooling \\(\\to\\) Dropout \\(\\to\\) FC-1000</li>
</ul>

<div class="env-block definition">
<strong>Global Average Pooling (GAP).</strong>
Instead of flattening the final feature maps and feeding them into large FC layers (as in AlexNet and VGG), GoogLeNet applies <em>global average pooling</em>: for each channel, compute the average over the entire spatial extent. This maps a \\(7 \\times 7 \\times 1024\\) tensor to a \\(1 \\times 1 \\times 1024\\) vector with <em>zero parameters</em>. GAP eliminates the parameter-heavy FC layers, reduces overfitting, and enforces the interpretation that each channel corresponds to a semantic feature.
</div>

<h3>Auxiliary Classifiers</h3>

<p>
GoogLeNet includes two <em>auxiliary classifiers</em> that branch off from intermediate Inception modules (4a and 4d). During training, their losses are weighted by 0.3 and added to the main loss. The purpose is to combat the vanishing gradient problem by injecting gradients into lower layers. At test time, these branches are discarded. (This technique was later superseded by batch normalization and residual connections.)
</p>

<div class="env-block remark">
<strong>The Name.</strong>
"GoogLeNet" is a deliberate homage to LeCun's "LeNet," with the capitalized "L" paying tribute to the original. The architecture is also known as "Inception v1" after the film Inception ("we need to go deeper"), and was followed by Inception v2, v3, and v4 with improvements like batch normalization, factorized convolutions, and residual connections.
</div>
`,
      visualizations: [
        {
          id: 'viz-inception-module',
          title: 'Inception Module Architecture',
          description: 'The four parallel branches process input at different scales. Channel counts shown for Inception 3a. Click branches to highlight data flow.',
          setup(container, controls) {
            const W = 780, H = 480;
            const viz = new VizEngine(container, { width: W, height: H, scale: 1, originX: 0, originY: 0 });
            const ctx = viz.ctx;

            let highlightBranch = -1;

            const branchColors = ['#58a6ff', '#3fb950', '#f0883e', '#bc8cff'];
            const branchNames = ['1x1 Conv', '3x3 Branch', '5x5 Branch', 'Pool Branch'];

            // Branch specs for Inception 3a
            const branches = [
              { label: '1x1 Conv',   steps: [{name: '1x1 Conv', ch: 64}],                          outCh: 64 },
              { label: '3x3 Branch', steps: [{name: '1x1 Conv', ch: 96}, {name: '3x3 Conv', ch: 128}], outCh: 128 },
              { label: '5x5 Branch', steps: [{name: '1x1 Conv', ch: 16}, {name: '5x5 Conv', ch: 32}],  outCh: 32 },
              { label: 'Pool Branch', steps: [{name: '3x3 MaxPool', ch: null}, {name: '1x1 Conv', ch: 32}], outCh: 32 }
            ];

            function draw() {
              ctx.fillStyle = viz.colors.bg;
              ctx.fillRect(0, 0, W, H);

              // Title
              ctx.fillStyle = viz.colors.white;
              ctx.font = 'bold 14px -apple-system,sans-serif';
              ctx.textAlign = 'center';
              ctx.textBaseline = 'top';
              ctx.fillText('Inception Module (Inception 3a)', W / 2, 10);

              // Previous Layer (input)
              const inputY = 60;
              const inputW = 500, inputH = 32;
              const inputX = (W - inputW) / 2;
              ctx.fillStyle = '#8b949e55';
              ctx.fillRect(inputX, inputY, inputW, inputH);
              ctx.strokeStyle = '#8b949e';
              ctx.lineWidth = 1.5;
              ctx.strokeRect(inputX, inputY, inputW, inputH);
              ctx.fillStyle = viz.colors.white;
              ctx.font = '12px -apple-system,sans-serif';
              ctx.textAlign = 'center';
              ctx.textBaseline = 'middle';
              ctx.fillText('Previous Layer Output: 28x28x192', W / 2, inputY + inputH / 2);

              // Branches
              const branchW = 110;
              const branchGap = 20;
              const totalBranchW = 4 * branchW + 3 * branchGap;
              const branchStartX = (W - totalBranchW) / 2;
              const stepH = 40;
              const stepGap = 15;
              const branchStartY = inputY + inputH + 50;

              // Draw connection lines from input to branches
              branches.forEach((branch, bi) => {
                const bx = branchStartX + bi * (branchW + branchGap) + branchW / 2;
                const color = highlightBranch === -1 || highlightBranch === bi
                  ? branchColors[bi] : branchColors[bi] + '33';
                ctx.strokeStyle = color;
                ctx.lineWidth = highlightBranch === bi ? 2.5 : 1.5;
                ctx.beginPath();
                ctx.moveTo(bx, inputY + inputH);
                ctx.lineTo(bx, branchStartY);
                ctx.stroke();

                // Arrow
                ctx.fillStyle = color;
                ctx.beginPath();
                ctx.moveTo(bx, branchStartY);
                ctx.lineTo(bx - 4, branchStartY - 8);
                ctx.lineTo(bx + 4, branchStartY - 8);
                ctx.closePath();
                ctx.fill();
              });

              // Draw each branch
              let maxBranchBottom = 0;
              branches.forEach((branch, bi) => {
                const bx = branchStartX + bi * (branchW + branchGap);
                const color = branchColors[bi];
                const dim = highlightBranch === -1 || highlightBranch === bi ? 1.0 : 0.3;
                const alphaFill = dim === 1.0 ? '66' : '22';
                const alphaStroke = dim === 1.0 ? 'cc' : '44';

                branch.steps.forEach((step, si) => {
                  const sy = branchStartY + si * (stepH + stepGap);
                  ctx.fillStyle = color + alphaFill;
                  ctx.strokeStyle = color + alphaStroke;
                  ctx.lineWidth = highlightBranch === bi ? 2 : 1;
                  ctx.beginPath();
                  ctx.roundRect(bx, sy, branchW, stepH, 6);
                  ctx.fill();
                  ctx.stroke();

                  ctx.fillStyle = dim === 1.0 ? viz.colors.white : viz.colors.white + '55';
                  ctx.font = '11px -apple-system,sans-serif';
                  ctx.textAlign = 'center';
                  ctx.textBaseline = 'middle';
                  ctx.fillText(step.name, bx + branchW / 2, sy + stepH / 2 - 6);
                  if (step.ch !== null) {
                    ctx.fillStyle = dim === 1.0 ? color : color + '55';
                    ctx.font = '10px -apple-system,sans-serif';
                    ctx.fillText(step.ch + ' channels', bx + branchW / 2, sy + stepH / 2 + 10);
                  }

                  // Connection within branch
                  if (si < branch.steps.length - 1) {
                    ctx.strokeStyle = color + alphaStroke;
                    ctx.lineWidth = 1;
                    ctx.beginPath();
                    ctx.moveTo(bx + branchW / 2, sy + stepH);
                    ctx.lineTo(bx + branchW / 2, sy + stepH + stepGap);
                    ctx.stroke();
                  }

                  maxBranchBottom = Math.max(maxBranchBottom, sy + stepH);
                });
              });

              // Concatenation bar
              const concatY = maxBranchBottom + 50;
              const concatH = 36;

              // Lines from branches to concat
              branches.forEach((branch, bi) => {
                const bx = branchStartX + bi * (branchW + branchGap) + branchW / 2;
                const lastStepY = branchStartY + (branch.steps.length - 1) * (stepH + stepGap) + stepH;
                const color = highlightBranch === -1 || highlightBranch === bi
                  ? branchColors[bi] : branchColors[bi] + '33';
                ctx.strokeStyle = color;
                ctx.lineWidth = highlightBranch === bi ? 2.5 : 1.5;
                ctx.beginPath();
                ctx.moveTo(bx, lastStepY);
                ctx.lineTo(bx, concatY);
                ctx.stroke();
              });

              // Concat box
              ctx.fillStyle = '#d2992266';
              ctx.strokeStyle = '#d29922';
              ctx.lineWidth = 2;
              ctx.beginPath();
              ctx.roundRect(inputX, concatY, inputW, concatH, 6);
              ctx.fill();
              ctx.stroke();

              ctx.fillStyle = viz.colors.white;
              ctx.font = 'bold 12px -apple-system,sans-serif';
              ctx.textAlign = 'center';
              ctx.textBaseline = 'middle';
              const totalCh = branches.reduce((s, b) => s + b.outCh, 0);
              ctx.fillText('Filter Concatenation: ' + branches.map(b => b.outCh).join(' + ') + ' = ' + totalCh + ' channels', W / 2, concatY + concatH / 2);

              // Output
              const outY = concatY + concatH + 20;
              ctx.fillStyle = viz.colors.text;
              ctx.font = '11px -apple-system,sans-serif';
              ctx.textAlign = 'center';
              ctx.fillText('Output: 28x28x' + totalCh, W / 2, outY);

              // Branch labels at top
              branches.forEach((branch, bi) => {
                const bx = branchStartX + bi * (branchW + branchGap);
                const dim = highlightBranch === -1 || highlightBranch === bi ? 1.0 : 0.3;
                ctx.fillStyle = dim === 1.0 ? branchColors[bi] : branchColors[bi] + '55';
                ctx.font = 'bold 10px -apple-system,sans-serif';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'bottom';
                ctx.fillText(branchNames[bi], bx + branchW / 2, branchStartY - 4);
              });
            }

            draw();

            // Click to highlight branches
            viz.canvas.addEventListener('click', (e) => {
              const rect = viz.canvas.getBoundingClientRect();
              const mx = (e.clientX - rect.left) * (W / rect.width);
              const my = (e.clientY - rect.top) * (H / rect.height);

              const branchW = 110;
              const branchGap = 20;
              const totalBranchW = 4 * branchW + 3 * branchGap;
              const branchStartX = (W - totalBranchW) / 2;

              let found = -1;
              for (let bi = 0; bi < 4; bi++) {
                const bx = branchStartX + bi * (branchW + branchGap);
                if (mx >= bx && mx <= bx + branchW && my >= 100 && my <= 380) {
                  found = bi;
                  break;
                }
              }

              highlightBranch = found === highlightBranch ? -1 : found;
              draw();
            });

            viz.canvas.addEventListener('mousemove', (e) => {
              const rect = viz.canvas.getBoundingClientRect();
              const mx = (e.clientX - rect.left) * (W / rect.width);
              const my = (e.clientY - rect.top) * (H / rect.height);

              const branchW = 110;
              const branchGap = 20;
              const totalBranchW = 4 * branchW + 3 * branchGap;
              const branchStartX = (W - totalBranchW) / 2;

              let onBranch = false;
              for (let bi = 0; bi < 4; bi++) {
                const bx = branchStartX + bi * (branchW + branchGap);
                if (mx >= bx && mx <= bx + branchW && my >= 100 && my <= 380) {
                  onBranch = true;
                  break;
                }
              }
              viz.canvas.style.cursor = onBranch ? 'pointer' : 'default';
            });

            return viz;
          }
        }
      ],
      exercises: [
        {
          question: 'In the Inception 3a module, the input is \\(28 \\times 28 \\times 192\\). The \\(3 \\times 3\\) branch first applies a \\(1 \\times 1\\) conv reducing to 96 channels, then a \\(3 \\times 3\\) conv producing 128 channels. Compute the multiply-add cost of this branch, and compare it to a direct \\(3 \\times 3\\) conv from 192 to 128 channels.',
          hint: 'For a conv layer: cost = \\(H \\times W \\times C_{out} \\times (k^2 \\times C_{in})\\).',
          solution: 'With bottleneck: \\(1 \\times 1\\) conv: \\(28^2 \\times 96 \\times 192 = 14.5\\text{M}\\). \\(3 \\times 3\\) conv: \\(28^2 \\times 128 \\times (9 \\times 96) = 86.5\\text{M}\\). Total: \\(101\\text{M}\\). Direct \\(3 \\times 3\\) conv: \\(28^2 \\times 128 \\times (9 \\times 192) = 173\\text{M}\\). The bottleneck reduces cost by about 42%.'
        },
        {
          question: 'GoogLeNet uses global average pooling instead of fully connected layers at the end. If the final feature map is \\(7 \\times 7 \\times 1024\\) and we have 1000 classes, compare the parameter count of (a) GAP + FC-1000 vs. (b) flattening + FC-4096 + FC-1000 (like AlexNet/VGG).',
          hint: 'GAP reduces \\(7 \\times 7 \\times 1024\\) to \\(1024\\) with zero parameters. FC-1000 then needs \\(1024 \\times 1000\\) parameters.',
          solution: '(a) GAP + FC-1000: \\(0 + 1024 \\times 1000 = 1{,}024{,}000\\) parameters. (b) Flatten + FC-4096 + FC-1000: \\(7 \\times 7 \\times 1024 \\times 4096 + 4096 \\times 1000 = 205{,}520{,}896 + 4{,}096{,}000 \\approx 209.6\\text{M}\\). GAP reduces classifier parameters by over 200x.'
        },
        {
          question: 'Why does the Inception module concatenate along the <em>channel</em> dimension rather than averaging or summing the branch outputs?',
          hint: 'Think about what happens to information when you average vs. concatenate.',
          solution: 'Concatenation preserves all information from every branch, allowing the next layer to learn how to best combine features at different scales. Averaging or summing would force all branches to have the same number of channels and would destroy information by collapsing distinct features into a single representation. Concatenation also lets the network assign different capacities (channel counts) to different branches based on what scales are most informative.'
        }
      ]
    },

    // ═══════════════════════════════════════════════════════════════════════════
    // SECTION 5: Architecture Evolution
    // ═══════════════════════════════════════════════════════════════════════════
    {
      id: 'ch10-sec05',
      title: '5. Architecture Evolution',
      content: `
<h2>Architecture Evolution: From LeNet to the Modern Era</h2>

<div class="env-block intuition">
<strong>Patterns in Progress.</strong>
Looking across LeNet-5, AlexNet, VGGNet, and GoogLeNet, clear evolutionary patterns emerge. Each generation refined ideas from its predecessors while introducing new principles. Understanding these trends prepares us for ResNet and beyond (Chapter 11).
</div>

<h3>Timeline and Key Milestones</h3>

<table style="width:100%;border-collapse:collapse;margin:1rem 0;">
  <tr style="background:#1a1a40;">
    <th style="padding:8px;border:1px solid #30363d;">Year</th>
    <th style="padding:8px;border:1px solid #30363d;">Network</th>
    <th style="padding:8px;border:1px solid #30363d;">Depth</th>
    <th style="padding:8px;border:1px solid #30363d;">Parameters</th>
    <th style="padding:8px;border:1px solid #30363d;">Top-5 Error</th>
    <th style="padding:8px;border:1px solid #30363d;">Key Innovation</th>
  </tr>
  <tr>
    <td style="padding:8px;border:1px solid #30363d;">1998</td>
    <td style="padding:8px;border:1px solid #30363d;">LeNet-5</td>
    <td style="padding:8px;border:1px solid #30363d;">7</td>
    <td style="padding:8px;border:1px solid #30363d;">60K</td>
    <td style="padding:8px;border:1px solid #30363d;">--</td>
    <td style="padding:8px;border:1px solid #30363d;">Conv + Pool template</td>
  </tr>
  <tr>
    <td style="padding:8px;border:1px solid #30363d;">2012</td>
    <td style="padding:8px;border:1px solid #30363d;">AlexNet</td>
    <td style="padding:8px;border:1px solid #30363d;">8</td>
    <td style="padding:8px;border:1px solid #30363d;">62.4M</td>
    <td style="padding:8px;border:1px solid #30363d;">15.3%</td>
    <td style="padding:8px;border:1px solid #30363d;">ReLU, Dropout, GPU</td>
  </tr>
  <tr>
    <td style="padding:8px;border:1px solid #30363d;">2014</td>
    <td style="padding:8px;border:1px solid #30363d;">VGG-16</td>
    <td style="padding:8px;border:1px solid #30363d;">16</td>
    <td style="padding:8px;border:1px solid #30363d;">138M</td>
    <td style="padding:8px;border:1px solid #30363d;">7.3%</td>
    <td style="padding:8px;border:1px solid #30363d;">Homogeneous 3x3</td>
  </tr>
  <tr>
    <td style="padding:8px;border:1px solid #30363d;">2014</td>
    <td style="padding:8px;border:1px solid #30363d;">GoogLeNet</td>
    <td style="padding:8px;border:1px solid #30363d;">22</td>
    <td style="padding:8px;border:1px solid #30363d;">6.8M</td>
    <td style="padding:8px;border:1px solid #30363d;">6.7%</td>
    <td style="padding:8px;border:1px solid #30363d;">Inception, 1x1 conv, GAP</td>
  </tr>
  <tr>
    <td style="padding:8px;border:1px solid #30363d;">2015</td>
    <td style="padding:8px;border:1px solid #30363d;">ResNet-152</td>
    <td style="padding:8px;border:1px solid #30363d;">152</td>
    <td style="padding:8px;border:1px solid #30363d;">60.2M</td>
    <td style="padding:8px;border:1px solid #30363d;">3.6%</td>
    <td style="padding:8px;border:1px solid #30363d;">Skip connections</td>
  </tr>
</table>

<div class="viz-placeholder" data-viz="viz-cnn-timeline"></div>

<h3>Evolutionary Trends</h3>

<h4>Trend 1: Increasing Depth</h4>
<p>
Network depth grew from 7 layers (LeNet) to 8 (AlexNet) to 16-19 (VGG) to 22 (GoogLeNet) to 152 (ResNet). Each leap required new techniques to enable training: ReLU (AlexNet), batch normalization (Inception v2), and skip connections (ResNet).
</p>

<h4>Trend 2: Smaller Kernels</h4>
<p>
Kernel sizes shrank from \\(11 \\times 11\\) and \\(5 \\times 5\\) (AlexNet) to exclusively \\(3 \\times 3\\) (VGG) and even \\(1 \\times 1\\) (GoogLeNet). VGG proved that stacking small kernels is strictly better than using large ones: same receptive field, fewer parameters, more non-linearities.
</p>

<h4>Trend 3: Replacing FC with GAP</h4>
<p>
Fully connected layers dominated early architectures (94% of AlexNet's parameters). GoogLeNet introduced global average pooling, reducing the classifier to a single FC layer with negligible parameters. This became standard in all subsequent architectures.
</p>

<h4>Trend 4: Efficient Computation</h4>
<p>
GoogLeNet's \\(1 \\times 1\\) bottleneck convolutions showed that <em>parameter efficiency</em> could coexist with depth. This principle would be extended by ResNet's bottleneck blocks, MobileNet's depthwise separable convolutions, and EfficientNet's compound scaling.
</p>

<h4>Trend 5: From Manual to Learned Architecture</h4>
<p>
The progression from hand-designed architectures (LeNet through ResNet) eventually led to neural architecture search (NAS) methods such as NASNet (Zoph et al., 2018) and EfficientNet (Tan & Le, 2019), where the architecture itself is learned by optimization. However, the design principles discovered by the architectures in this chapter remain foundational.
</p>

<div class="env-block remark">
<strong>What Comes Next.</strong>
Chapter 11 covers ResNet and its successors (DenseNet, SE-Net, EfficientNet). The key innovation of ResNet, the skip connection, solved the degradation problem that limited VGG-style networks to about 20 layers, enabling networks hundreds of layers deep and fundamentally changing how we think about depth.
</div>

<h3>Summary of Key Principles</h3>

<div class="env-block definition">
<strong>Principles of CNN Architecture Design (circa 2015).</strong>
<ol>
  <li><strong>Use small convolutions</strong> (\\(3 \\times 3\\)) and stack them to achieve large receptive fields.</li>
  <li><strong>Double channels when halving spatial dimensions</strong> to maintain roughly constant computational cost per layer.</li>
  <li><strong>Use 1x1 convolutions</strong> for channel-wise dimensionality reduction before expensive operations.</li>
  <li><strong>Replace FC layers with global average pooling</strong> to reduce parameters and overfitting.</li>
  <li><strong>Apply batch normalization</strong> (or its successors) to stabilize training of deep networks.</li>
  <li><strong>Multi-scale processing</strong> (Inception) captures features at different spatial resolutions.</li>
</ol>
</div>
`,
      visualizations: [
        {
          id: 'viz-cnn-timeline',
          title: 'CNN Architecture Evolution: Accuracy vs. Depth vs. Parameters',
          description: 'Bubble chart showing the evolution of CNN architectures. Bubble size represents parameter count. Hover for details.',
          setup(container, controls) {
            const W = 800, H = 450;
            const viz = new VizEngine(container, { width: W, height: H, scale: 1, originX: 0, originY: 0 });
            const ctx = viz.ctx;

            const architectures = [
              { name: 'LeNet-5',    year: 1998, error: 50,   params: 0.06,  depth: 7,   color: '#8b949e', innovation: 'Conv + Pool template' },
              { name: 'AlexNet',    year: 2012, error: 15.3, params: 62.4,  depth: 8,   color: '#58a6ff', innovation: 'ReLU, Dropout, GPU training' },
              { name: 'ZFNet',      year: 2013, error: 11.2, params: 62.4,  depth: 8,   color: '#3fb9a0', innovation: 'Deconv visualization' },
              { name: 'VGG-16',     year: 2014, error: 7.3,  params: 138,   depth: 16,  color: '#f0883e', innovation: 'Homogeneous 3x3 convs' },
              { name: 'VGG-19',     year: 2014, error: 7.3,  params: 144,   depth: 19,  color: '#f0883e', innovation: '3 extra conv layers' },
              { name: 'GoogLeNet',  year: 2014, error: 6.7,  params: 6.8,   depth: 22,  color: '#bc8cff', innovation: 'Inception module, 1x1 conv, GAP' },
              { name: 'ResNet-50',  year: 2015, error: 3.6,  params: 25.6,  depth: 50,  color: '#f85149', innovation: 'Skip connections' },
              { name: 'ResNet-152', year: 2015, error: 3.6,  params: 60.2,  depth: 152, color: '#f85149', innovation: '152 layers deep' },
              { name: 'Inception v3', year: 2015, error: 3.5, params: 23.8, depth: 48,  color: '#d29922', innovation: 'Factorized convolutions' }
            ];

            // Chart area
            const padL = 70, padR = 40, padT = 55, padB = 75;
            const chartW = W - padL - padR;
            const chartH = H - padT - padB;

            // Scales
            const yearMin = 1996, yearMax = 2017;
            const errorMin = 0, errorMax = 55;

            function xScale(year) { return padL + (year - yearMin) / (yearMax - yearMin) * chartW; }
            function yScale(err)  { return padT + (err - errorMin) / (errorMax - errorMin) * chartH; }
            function radiusScale(params) { return Math.max(8, Math.sqrt(params) * 3.2); }

            let hoveredIdx = -1;

            function draw() {
              ctx.fillStyle = viz.colors.bg;
              ctx.fillRect(0, 0, W, H);

              // Title
              ctx.fillStyle = viz.colors.white;
              ctx.font = 'bold 14px -apple-system,sans-serif';
              ctx.textAlign = 'center';
              ctx.textBaseline = 'top';
              ctx.fillText('CNN Architecture Evolution (1998-2015)', W / 2, 10);
              ctx.fillStyle = viz.colors.text;
              ctx.font = '11px -apple-system,sans-serif';
              ctx.fillText('Bubble size = parameter count  |  Lower = better accuracy', W / 2, 30);

              // Grid lines
              ctx.strokeStyle = '#1a1a40';
              ctx.lineWidth = 0.5;
              for (let e = 0; e <= 50; e += 10) {
                const y = yScale(e);
                ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
              }
              for (let yr = 1998; yr <= 2016; yr += 2) {
                const x = xScale(yr);
                ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, padT + chartH); ctx.stroke();
              }

              // Axes
              ctx.strokeStyle = '#4a4a7a';
              ctx.lineWidth = 1.5;
              ctx.beginPath();
              ctx.moveTo(padL, padT);
              ctx.lineTo(padL, padT + chartH);
              ctx.lineTo(W - padR, padT + chartH);
              ctx.stroke();

              // Y-axis labels
              ctx.fillStyle = viz.colors.text;
              ctx.font = '10px -apple-system,sans-serif';
              ctx.textAlign = 'right';
              ctx.textBaseline = 'middle';
              for (let e = 0; e <= 50; e += 10) {
                ctx.fillText(e + '%', padL - 8, yScale(e));
              }

              // Y-axis title
              ctx.save();
              ctx.translate(16, padT + chartH / 2);
              ctx.rotate(-Math.PI / 2);
              ctx.fillStyle = viz.colors.text;
              ctx.font = '11px -apple-system,sans-serif';
              ctx.textAlign = 'center';
              ctx.fillText('Top-5 Error Rate (%)', 0, 0);
              ctx.restore();

              // X-axis labels
              ctx.textAlign = 'center';
              ctx.textBaseline = 'top';
              for (let yr = 1998; yr <= 2016; yr += 2) {
                ctx.fillText(yr.toString(), xScale(yr), padT + chartH + 6);
              }
              ctx.font = '11px -apple-system,sans-serif';
              ctx.fillText('Year', W / 2, padT + chartH + 24);

              // Human performance line
              const humanY = yScale(5.1);
              ctx.strokeStyle = '#3fb950';
              ctx.lineWidth = 1;
              ctx.setLineDash([6, 4]);
              ctx.beginPath();
              ctx.moveTo(padL, humanY);
              ctx.lineTo(W - padR, humanY);
              ctx.stroke();
              ctx.setLineDash([]);
              ctx.fillStyle = '#3fb950';
              ctx.font = '10px -apple-system,sans-serif';
              ctx.textAlign = 'left';
              ctx.fillText('Human (~5.1%)', padL + 4, humanY - 14);

              // Draw bubbles (draw hovered last for z-order)
              const drawOrder = architectures.map((_, i) => i).sort((a, b) => {
                if (a === hoveredIdx) return 1;
                if (b === hoveredIdx) return -1;
                return architectures[b].params - architectures[a].params;
              });

              drawOrder.forEach(i => {
                const arch = architectures[i];
                const bx = xScale(arch.year);
                const by = yScale(arch.error);
                const br = radiusScale(arch.params);
                const isHovered = i === hoveredIdx;

                // Glow
                if (isHovered) {
                  ctx.shadowColor = arch.color;
                  ctx.shadowBlur = 15;
                }

                ctx.fillStyle = arch.color + (isHovered ? '99' : '55');
                ctx.beginPath();
                ctx.arc(bx, by, br, 0, Math.PI * 2);
                ctx.fill();

                ctx.strokeStyle = arch.color;
                ctx.lineWidth = isHovered ? 2.5 : 1.5;
                ctx.beginPath();
                ctx.arc(bx, by, br, 0, Math.PI * 2);
                ctx.stroke();

                ctx.shadowColor = 'transparent';
                ctx.shadowBlur = 0;

                // Label
                ctx.fillStyle = isHovered ? viz.colors.white : viz.colors.text;
                ctx.font = (isHovered ? 'bold ' : '') + '10px -apple-system,sans-serif';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'bottom';

                // Offset labels to avoid overlap
                let labelY = by - br - 5;
                let labelX = bx;
                if (arch.name === 'VGG-19') { labelX += 35; labelY += 10; }
                if (arch.name === 'ResNet-152') { labelX += 30; }

                ctx.fillText(arch.name, labelX, labelY);
              });

              // Info panel on hover
              if (hoveredIdx >= 0) {
                const arch = architectures[hoveredIdx];
                const px = 30, py = padT + chartH - 100;
                ctx.fillStyle = '#14142edd';
                ctx.strokeStyle = arch.color;
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.roundRect(px, py, 310, 95, 8);
                ctx.fill();
                ctx.stroke();

                ctx.fillStyle = arch.color;
                ctx.font = 'bold 13px -apple-system,sans-serif';
                ctx.textAlign = 'left';
                ctx.textBaseline = 'top';
                ctx.fillText(arch.name + ' (' + arch.year + ')', px + 12, py + 10);

                ctx.fillStyle = viz.colors.text;
                ctx.font = '11px -apple-system,sans-serif';
                ctx.fillText('Depth: ' + arch.depth + ' layers  |  Params: ' + arch.params + 'M', px + 12, py + 30);
                ctx.fillText('Top-5 Error: ' + (arch.error < 50 ? arch.error + '%' : 'N/A (MNIST)'), px + 12, py + 48);
                ctx.fillText('Innovation: ' + arch.innovation, px + 12, py + 66);
              }
            }

            draw();

            viz.canvas.addEventListener('mousemove', (e) => {
              const rect = viz.canvas.getBoundingClientRect();
              const mx = (e.clientX - rect.left) * (W / rect.width);
              const my = (e.clientY - rect.top) * (H / rect.height);

              let found = -1;
              let minDist = Infinity;
              architectures.forEach((arch, i) => {
                const bx = xScale(arch.year);
                const by = yScale(arch.error);
                const br = radiusScale(arch.params);
                const d = Math.sqrt((mx - bx) ** 2 + (my - by) ** 2);
                if (d <= br + 5 && d < minDist) {
                  found = i;
                  minDist = d;
                }
              });
              if (found !== hoveredIdx) {
                hoveredIdx = found;
                viz.canvas.style.cursor = found >= 0 ? 'pointer' : 'default';
                draw();
              }
            });

            viz.canvas.addEventListener('mouseleave', () => {
              hoveredIdx = -1;
              draw();
            });

            return viz;
          }
        }
      ],
      exercises: [
        {
          question: 'Consider the trend from AlexNet (62.4M params, 15.3% error) to GoogLeNet (6.8M params, 6.7% error). GoogLeNet achieves lower error with 9x fewer parameters. What architectural techniques account for most of this parameter reduction?',
          hint: 'Think about what contributes most to AlexNet\'s parameter count and how GoogLeNet avoids that.',
          solution: 'Three main techniques: (1) <strong>Global average pooling</strong> replaces the massive FC layers (which account for 94% of AlexNet\'s params). (2) <strong>1x1 bottleneck convolutions</strong> reduce the channel dimension before expensive 3x3 and 5x5 operations. (3) <strong>Factored multi-scale design</strong> (Inception modules) allocates capacity efficiently across scales rather than using uniformly wide layers. Together, these reduce the parameter count by 9x while actually increasing the network\'s effective depth and representational power.'
        },
        {
          question: 'Why is depth so important for CNN performance? Provide an argument based on the <em>representational hierarchy</em> that CNNs learn.',
          hint: 'Think about what different layers learn in a CNN and how composition of simple features builds complex detectors.',
          solution: 'Deep CNNs learn a hierarchical decomposition of visual patterns. Early layers detect edges and textures, middle layers combine them into parts (eyes, wheels, stripes), and deep layers compose parts into whole objects (faces, cars, animals). Each layer of abstraction builds on the previous one through non-linear composition. Shallow networks would need exponentially many features to directly represent the same complex patterns that a deep network builds incrementally. This compositional hierarchy mirrors the structure of natural images, where objects are composed of parts, parts of sub-parts, and so on. More depth means more levels of abstraction and more efficient representation of complex visual concepts.'
        },
        {
          question: 'A common design pattern is to double the number of channels each time the spatial dimensions are halved by pooling. Explain the computational rationale behind this rule.',
          hint: 'Consider the FLOP count of a convolution layer: \\(H \\times W \\times C_{out} \\times (k^2 \\times C_{in})\\). What happens when \\(H, W\\) are halved and \\(C\\) is doubled?',
          solution: 'If we halve spatial dimensions (\\(H/2 \\times W/2\\)) and double channels (\\(2C\\)), the computational cost per conv layer becomes \\((H/2)(W/2)(2C)(k^2 \\cdot 2C) = HWC \\cdot k^2 C\\), which is the same as before. This "doubling rule" maintains roughly constant computational cost per layer throughout the network, ensuring that no single layer becomes a bottleneck or is underutilized. It also balances spatial resolution (where the network localizes features) against channel capacity (the vocabulary of features the network can detect).'
        }
      ]
    }
  ]
});
