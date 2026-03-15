window.CHAPTERS = window.CHAPTERS || [];
window.CHAPTERS.push({
    id: 'ch08',
    number: 8,
    title: 'Regularization Techniques',
    subtitle: 'Controlling Complexity to Improve Generalization',
    sections: [

        // ===== SECTION 1: Bias-Variance Tradeoff =====
        {
            id: 'bias-variance',
            title: 'Bias-Variance Tradeoff',
            content: `
                <h2>The Fundamental Tension in Learning</h2>

                <div class="env-block intuition">
                    <div class="env-title">Why Do Models Fail?</div>
                    <div class="env-body">
                        <p>Every supervised learning model faces a core dilemma: it must be flexible enough to capture the true pattern in the data, yet constrained enough to avoid memorizing noise. Too simple, and the model misses the signal (<em>underfitting</em>). Too complex, and the model memorizes the training set but fails on unseen data (<em>overfitting</em>). Regularization is the art of navigating this tension.</p>
                    </div>
                </div>

                <h3>The Bias-Variance Decomposition</h3>

                <p>Consider a target function \\(f(x)\\) corrupted by noise \\(\\epsilon\\) with \\(\\mathbb{E}[\\epsilon]=0\\) and \\(\\text{Var}(\\epsilon) = \\sigma^2\\). For a model \\(\\hat{f}(x)\\) trained on a dataset \\(D\\), the expected prediction error at a point \\(x\\) decomposes as:</p>

                <div class="env-block theorem">
                    <div class="env-title">Bias-Variance Decomposition</div>
                    <div class="env-body">
                        <p>For squared loss, the expected test error decomposes into three additive terms:</p>
                        \\[
                        \\mathbb{E}_D\\!\\left[(y - \\hat{f}(x))^2\\right]
                        = \\underbrace{\\bigl(f(x) - \\mathbb{E}_D[\\hat{f}(x)]\\bigr)^2}_{\\text{Bias}^2}
                        + \\underbrace{\\mathbb{E}_D\\!\\left[(\\hat{f}(x) - \\mathbb{E}_D[\\hat{f}(x)])^2\\right]}_{\\text{Variance}}
                        + \\underbrace{\\sigma^2}_{\\text{Irreducible noise}}
                        \\]
                        <p>Bias measures how far the average prediction is from the truth. Variance measures how much predictions fluctuate across different training sets. The noise term \\(\\sigma^2\\) is a lower bound that no model can beat.</p>
                    </div>
                </div>

                <div class="env-block proof">
                    <div class="env-title">Derivation</div>
                    <div class="env-body">
                        <p>Write \\(y = f(x) + \\epsilon\\). Let \\(\\bar{f}(x) = \\mathbb{E}_D[\\hat{f}(x)]\\). Then:</p>
                        \\[
                        \\mathbb{E}[(y - \\hat{f})^2]
                        = \\mathbb{E}[(f + \\epsilon - \\hat{f})^2]
                        = \\mathbb{E}[(f - \\hat{f})^2] + \\sigma^2
                        \\]
                        <p>since \\(\\epsilon\\) is independent of \\(\\hat{f}\\) with zero mean. Now expand the first term by adding and subtracting \\(\\bar{f}\\):</p>
                        \\[
                        \\mathbb{E}[(f - \\hat{f})^2]
                        = \\mathbb{E}[(f - \\bar{f} + \\bar{f} - \\hat{f})^2]
                        = (f - \\bar{f})^2 + \\mathbb{E}[(\\hat{f} - \\bar{f})^2]
                        \\]
                        <p>The cross term vanishes because \\(\\mathbb{E}[\\hat{f} - \\bar{f}] = 0\\). This gives the three-term decomposition. <span class="qed">\\(\\blacksquare\\)</span></p>
                    </div>
                </div>

                <h3>Model Complexity and the U-Shaped Curve</h3>

                <p>As model complexity increases (for example, polynomial degree, number of hidden units, or tree depth):</p>
                <ul>
                    <li><strong>Bias decreases:</strong> A more flexible model can approximate \\(f(x)\\) more closely.</li>
                    <li><strong>Variance increases:</strong> A more flexible model is more sensitive to the specific training data.</li>
                    <li><strong>Training error decreases monotonically</strong> (the model fits the data better).</li>
                    <li><strong>Test error follows a U-shape,</strong> first decreasing as bias drops, then increasing as variance dominates.</li>
                </ul>

                <p>The sweet spot, where test error is minimized, represents the optimal model complexity. <em>Regularization techniques shift this sweet spot, allowing us to use high-capacity models without overfitting.</em></p>

                <div class="viz-placeholder" data-viz="viz-bias-variance"></div>

                <div class="env-block definition">
                    <div class="env-title">Underfitting and Overfitting</div>
                    <div class="env-body">
                        <p><strong>Underfitting</strong> occurs when a model is too simple to capture the underlying structure. Both training and test errors are high. The model has high bias and low variance.</p>
                        <p><strong>Overfitting</strong> occurs when a model is so complex that it fits the noise in the training data. Training error is low but test error is high. The model has low bias and high variance.</p>
                    </div>
                </div>

                <div class="env-block remark">
                    <div class="env-title">The Double-Descent Phenomenon</div>
                    <div class="env-body">
                        <p>Modern deep networks with more parameters than training examples sometimes exhibit <em>double descent</em>: test error decreases, increases, then decreases again beyond the interpolation threshold. This challenges the classical U-shaped picture but does not invalidate the bias-variance decomposition; it rather reveals that the implicit regularization of gradient descent can keep variance controlled even in over-parameterized regimes.</p>
                    </div>
                </div>

                <div class="env-block example">
                    <div class="env-title">Example: Polynomial Regression</div>
                    <div class="env-body">
                        <p>Fitting a degree-1 polynomial (a line) to a cubic target \\(f(x) = x^3 - 2x\\) gives high bias (the line cannot capture curvature). A degree-15 polynomial can fit any 16 data points perfectly, but its predictions oscillate wildly between the points (high variance). A degree-3 polynomial is just right: low bias and controlled variance.</p>
                    </div>
                </div>
            `,
            visualizations: [
                {
                    id: 'viz-bias-variance',
                    title: 'Bias-Variance Tradeoff: Polynomial Fitting',
                    description: 'Adjust polynomial degree to see underfitting, optimal fit, and overfitting. The right panel shows how training error (blue) always decreases with complexity, while test error (red) follows a U-shape.',
                    setup: function(body, controls) {
                        var viz = new VizEngine(body, { width: 800, height: 400, scale: 1, originX: 0, originY: 0 });
                        var ctx = viz.ctx;
                        var W = viz.width, H = viz.height;
                        var degree = 3;
                        var seed = 42;

                        // Seeded random
                        function seededRandom(s) {
                            var x = Math.sin(s) * 10000;
                            return x - Math.floor(x);
                        }

                        // True function: sin(x) scaled
                        function trueF(x) {
                            return Math.sin(1.5 * x) * 1.5 + 0.5 * x;
                        }

                        // Generate training data
                        var nTrain = 15;
                        var nTest = 200;
                        var trainX = [], trainY = [];
                        for (var i = 0; i < nTrain; i++) {
                            var x = -2.5 + 5 * i / (nTrain - 1);
                            var noise = (seededRandom(seed + i * 7) - 0.5) * 1.5;
                            trainX.push(x);
                            trainY.push(trueF(x) + noise);
                        }

                        // Polynomial fit using normal equations
                        function polyFit(xs, ys, deg) {
                            var n = xs.length;
                            var d = deg + 1;
                            // Build Vandermonde
                            var X = [];
                            for (var i = 0; i < n; i++) {
                                var row = [];
                                for (var j = 0; j < d; j++) {
                                    row.push(Math.pow(xs[i], j));
                                }
                                X.push(row);
                            }
                            // X^T X
                            var XtX = [];
                            for (var i = 0; i < d; i++) {
                                XtX.push([]);
                                for (var j = 0; j < d; j++) {
                                    var s = 0;
                                    for (var k = 0; k < n; k++) s += X[k][i] * X[k][j];
                                    // Ridge tiny regularization for numerical stability
                                    XtX[i].push(s + (i === j ? 1e-8 : 0));
                                }
                            }
                            // X^T y
                            var Xty = [];
                            for (var i = 0; i < d; i++) {
                                var s = 0;
                                for (var k = 0; k < n; k++) s += X[k][i] * ys[k];
                                Xty.push(s);
                            }
                            // Solve via Gaussian elimination
                            var A = XtX.map(function(r, i) { return r.concat([Xty[i]]); });
                            for (var col = 0; col < d; col++) {
                                var maxRow = col;
                                for (var row = col + 1; row < d; row++) {
                                    if (Math.abs(A[row][col]) > Math.abs(A[maxRow][col])) maxRow = row;
                                }
                                var tmp = A[col]; A[col] = A[maxRow]; A[maxRow] = tmp;
                                if (Math.abs(A[col][col]) < 1e-14) continue;
                                for (var row = col + 1; row < d; row++) {
                                    var f = A[row][col] / A[col][col];
                                    for (var j = col; j <= d; j++) A[row][j] -= f * A[col][j];
                                }
                            }
                            var coeff = new Array(d);
                            for (var i = d - 1; i >= 0; i--) {
                                coeff[i] = A[i][d];
                                for (var j = i + 1; j < d; j++) coeff[i] -= A[i][j] * coeff[j];
                                coeff[i] /= A[i][i];
                            }
                            return coeff;
                        }

                        function polyEval(coeff, x) {
                            var y = 0;
                            for (var i = 0; i < coeff.length; i++) y += coeff[i] * Math.pow(x, i);
                            return y;
                        }

                        // Compute errors for all degrees
                        function computeErrors() {
                            var trainErrors = [], testErrors = [];
                            for (var deg = 1; deg <= 12; deg++) {
                                var coeff = polyFit(trainX, trainY, deg);
                                // Train MSE
                                var mse = 0;
                                for (var i = 0; i < nTrain; i++) {
                                    var diff = trainY[i] - polyEval(coeff, trainX[i]);
                                    mse += diff * diff;
                                }
                                trainErrors.push(mse / nTrain);
                                // Test MSE (use true function + fresh noise)
                                var testMse = 0;
                                for (var i = 0; i < nTest; i++) {
                                    var x = -2.5 + 5 * i / (nTest - 1);
                                    var yTrue = trueF(x);
                                    var yPred = polyEval(coeff, x);
                                    testMse += (yTrue - yPred) * (yTrue - yPred);
                                }
                                testErrors.push(testMse / nTest);
                            }
                            return { train: trainErrors, test: testErrors };
                        }

                        var allErrors = computeErrors();

                        function draw() {
                            viz.clear();
                            var leftW = W * 0.52;
                            var rightW = W - leftW;

                            // === LEFT PANEL: Data + Fit ===
                            var margin = { top: 40, bottom: 40, left: 50, right: 20 };
                            var plotW = leftW - margin.left - margin.right;
                            var plotH = H - margin.top - margin.bottom;
                            var xMin = -3, xMax = 3, yMin = -5, yMax = 5;

                            function toPixelL(x, y) {
                                var px = margin.left + (x - xMin) / (xMax - xMin) * plotW;
                                var py = margin.top + (1 - (y - yMin) / (yMax - yMin)) * plotH;
                                return [px, py];
                            }

                            // Grid
                            ctx.strokeStyle = viz.colors.grid;
                            ctx.lineWidth = 0.5;
                            for (var gx = Math.ceil(xMin); gx <= xMax; gx++) {
                                var p = toPixelL(gx, 0);
                                ctx.beginPath(); ctx.moveTo(p[0], margin.top); ctx.lineTo(p[0], margin.top + plotH); ctx.stroke();
                            }
                            for (var gy = Math.ceil(yMin); gy <= yMax; gy++) {
                                var p = toPixelL(0, gy);
                                ctx.beginPath(); ctx.moveTo(margin.left, p[1]); ctx.lineTo(margin.left + plotW, p[1]); ctx.stroke();
                            }

                            // Axes
                            ctx.strokeStyle = viz.colors.axis;
                            ctx.lineWidth = 1;
                            ctx.beginPath(); ctx.moveTo(margin.left, margin.top + plotH); ctx.lineTo(margin.left + plotW, margin.top + plotH); ctx.stroke();
                            ctx.beginPath(); ctx.moveTo(margin.left, margin.top); ctx.lineTo(margin.left, margin.top + plotH); ctx.stroke();

                            // True function (dashed)
                            ctx.strokeStyle = viz.colors.green;
                            ctx.lineWidth = 2;
                            ctx.setLineDash([6, 4]);
                            ctx.beginPath();
                            for (var i = 0; i <= 200; i++) {
                                var x = xMin + (xMax - xMin) * i / 200;
                                var y = trueF(x);
                                var p = toPixelL(x, y);
                                if (i === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]);
                            }
                            ctx.stroke();
                            ctx.setLineDash([]);

                            // Polynomial fit
                            var coeff = polyFit(trainX, trainY, degree);
                            ctx.strokeStyle = viz.colors.blue;
                            ctx.lineWidth = 2.5;
                            ctx.beginPath();
                            for (var i = 0; i <= 300; i++) {
                                var x = xMin + (xMax - xMin) * i / 300;
                                var y = polyEval(coeff, x);
                                y = Math.max(yMin - 1, Math.min(yMax + 1, y));
                                var p = toPixelL(x, y);
                                if (i === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]);
                            }
                            ctx.stroke();

                            // Data points
                            for (var i = 0; i < nTrain; i++) {
                                var p = toPixelL(trainX[i], trainY[i]);
                                ctx.fillStyle = viz.colors.orange;
                                ctx.beginPath(); ctx.arc(p[0], p[1], 4, 0, Math.PI * 2); ctx.fill();
                                ctx.strokeStyle = '#000'; ctx.lineWidth = 0.5; ctx.stroke();
                            }

                            // Label
                            ctx.fillStyle = viz.colors.white;
                            ctx.font = 'bold 14px -apple-system,sans-serif';
                            ctx.textAlign = 'left';
                            ctx.fillText('Degree ' + degree + ' polynomial', margin.left + 5, margin.top - 10);

                            // Status
                            var status = degree <= 2 ? 'UNDERFITTING' : (degree >= 8 ? 'OVERFITTING' : 'GOOD FIT');
                            var statusColor = degree <= 2 ? viz.colors.red : (degree >= 8 ? viz.colors.red : viz.colors.green);
                            ctx.fillStyle = statusColor;
                            ctx.font = 'bold 12px -apple-system,sans-serif';
                            ctx.textAlign = 'right';
                            ctx.fillText(status, margin.left + plotW, margin.top - 10);

                            // Legend
                            ctx.font = '11px -apple-system,sans-serif';
                            var legY = margin.top + 15;
                            ctx.fillStyle = viz.colors.green;
                            ctx.fillRect(margin.left + 8, legY - 5, 16, 3);
                            ctx.fillStyle = viz.colors.text;
                            ctx.textAlign = 'left';
                            ctx.fillText('True f(x)', margin.left + 30, legY);
                            ctx.fillStyle = viz.colors.blue;
                            ctx.fillRect(margin.left + 8, legY + 13, 16, 3);
                            ctx.fillStyle = viz.colors.text;
                            ctx.fillText('Fitted polynomial', margin.left + 30, legY + 18);
                            ctx.fillStyle = viz.colors.orange;
                            ctx.beginPath(); ctx.arc(margin.left + 16, legY + 34, 3, 0, Math.PI * 2); ctx.fill();
                            ctx.fillStyle = viz.colors.text;
                            ctx.fillText('Training data', margin.left + 30, legY + 36);

                            // === RIGHT PANEL: Error Curves ===
                            var rMargin = { top: 40, bottom: 40, left: 40, right: 25 };
                            var rPlotW = rightW - rMargin.left - rMargin.right;
                            var rPlotH = H - rMargin.top - rMargin.bottom;
                            var rOffX = leftW;

                            // Find y range for errors
                            var maxErr = 0;
                            for (var i = 0; i < allErrors.train.length; i++) {
                                maxErr = Math.max(maxErr, allErrors.train[i], allErrors.test[i]);
                            }
                            maxErr = Math.min(maxErr, 15);

                            function toPixelR(deg, err) {
                                var px = rOffX + rMargin.left + (deg - 1) / 11 * rPlotW;
                                var py = rMargin.top + (1 - err / maxErr) * rPlotH;
                                return [px, py];
                            }

                            // Grid
                            ctx.strokeStyle = viz.colors.grid;
                            ctx.lineWidth = 0.5;
                            for (var d = 1; d <= 12; d++) {
                                var p = toPixelR(d, 0);
                                ctx.beginPath(); ctx.moveTo(p[0], rMargin.top); ctx.lineTo(p[0], rMargin.top + rPlotH); ctx.stroke();
                            }

                            // Axes
                            ctx.strokeStyle = viz.colors.axis;
                            ctx.lineWidth = 1;
                            ctx.beginPath(); ctx.moveTo(rOffX + rMargin.left, rMargin.top + rPlotH); ctx.lineTo(rOffX + rMargin.left + rPlotW, rMargin.top + rPlotH); ctx.stroke();
                            ctx.beginPath(); ctx.moveTo(rOffX + rMargin.left, rMargin.top); ctx.lineTo(rOffX + rMargin.left, rMargin.top + rPlotH); ctx.stroke();

                            // X labels
                            ctx.fillStyle = viz.colors.text;
                            ctx.font = '10px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            for (var d = 1; d <= 12; d += 1) {
                                var p = toPixelR(d, 0);
                                ctx.fillText(d, p[0], rMargin.top + rPlotH + 14);
                            }
                            ctx.fillText('Polynomial Degree', rOffX + rMargin.left + rPlotW / 2, rMargin.top + rPlotH + 30);

                            // Y label
                            ctx.save();
                            ctx.translate(rOffX + 12, rMargin.top + rPlotH / 2);
                            ctx.rotate(-Math.PI / 2);
                            ctx.fillStyle = viz.colors.text;
                            ctx.font = '10px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText('MSE', 0, 0);
                            ctx.restore();

                            // Training error curve
                            ctx.strokeStyle = viz.colors.blue;
                            ctx.lineWidth = 2;
                            ctx.beginPath();
                            for (var i = 0; i < allErrors.train.length; i++) {
                                var p = toPixelR(i + 1, Math.min(allErrors.train[i], maxErr));
                                if (i === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]);
                            }
                            ctx.stroke();

                            // Test error curve
                            ctx.strokeStyle = viz.colors.red;
                            ctx.lineWidth = 2;
                            ctx.beginPath();
                            for (var i = 0; i < allErrors.test.length; i++) {
                                var p = toPixelR(i + 1, Math.min(allErrors.test[i], maxErr));
                                if (i === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]);
                            }
                            ctx.stroke();

                            // Dots on curves
                            for (var i = 0; i < allErrors.train.length; i++) {
                                var pt = toPixelR(i + 1, Math.min(allErrors.train[i], maxErr));
                                ctx.fillStyle = viz.colors.blue;
                                ctx.beginPath(); ctx.arc(pt[0], pt[1], 3, 0, Math.PI * 2); ctx.fill();
                                var pe = toPixelR(i + 1, Math.min(allErrors.test[i], maxErr));
                                ctx.fillStyle = viz.colors.red;
                                ctx.beginPath(); ctx.arc(pe[0], pe[1], 3, 0, Math.PI * 2); ctx.fill();
                            }

                            // Current degree indicator
                            var cpt = toPixelR(degree, 0);
                            ctx.strokeStyle = viz.colors.yellow;
                            ctx.lineWidth = 1.5;
                            ctx.setLineDash([4, 3]);
                            ctx.beginPath(); ctx.moveTo(cpt[0], rMargin.top); ctx.lineTo(cpt[0], rMargin.top + rPlotH); ctx.stroke();
                            ctx.setLineDash([]);

                            // Title and legend
                            ctx.fillStyle = viz.colors.white;
                            ctx.font = 'bold 14px -apple-system,sans-serif';
                            ctx.textAlign = 'left';
                            ctx.fillText('Error vs Complexity', rOffX + rMargin.left + 5, rMargin.top - 10);

                            ctx.font = '11px -apple-system,sans-serif';
                            var rlY = rMargin.top + 15;
                            ctx.fillStyle = viz.colors.blue;
                            ctx.fillRect(rOffX + rMargin.left + 8, rlY - 5, 16, 3);
                            ctx.fillStyle = viz.colors.text;
                            ctx.fillText('Train error', rOffX + rMargin.left + 30, rlY);
                            ctx.fillStyle = viz.colors.red;
                            ctx.fillRect(rOffX + rMargin.left + 8, rlY + 13, 16, 3);
                            ctx.fillStyle = viz.colors.text;
                            ctx.fillText('Test error', rOffX + rMargin.left + 30, rlY + 18);
                        }

                        draw();

                        VizEngine.createSlider(controls, 'Degree', 1, 12, 3, 1, function(val) {
                            degree = Math.round(val);
                            draw();
                        });

                        VizEngine.createButton(controls, 'New Data', function() {
                            seed = Math.floor(Math.random() * 10000);
                            trainX = []; trainY = [];
                            for (var i = 0; i < nTrain; i++) {
                                var x = -2.5 + 5 * i / (nTrain - 1);
                                var noise = (seededRandom(seed + i * 7) - 0.5) * 1.5;
                                trainX.push(x);
                                trainY.push(trueF(x) + noise);
                            }
                            allErrors = computeErrors();
                            draw();
                        });
                    }
                }
            ],
            exercises: [
                {
                    question: 'In the bias-variance decomposition \\(\\mathbb{E}[(y - \\hat{f})^2] = \\text{Bias}^2 + \\text{Variance} + \\sigma^2\\), explain why the irreducible noise \\(\\sigma^2\\) sets a floor on test error that no model can beat, regardless of complexity.',
                    hint: 'Think about what \\(\\sigma^2\\) represents: it is the variance of \\(\\epsilon\\) in \\(y = f(x) + \\epsilon\\). Even if you know \\(f(x)\\) exactly, what error remains?',
                    solution: 'Even a perfect model that recovers \\(f(x)\\) exactly still suffers error \\(\\sigma^2\\) because \\(y = f(x) + \\epsilon\\) and the noise \\(\\epsilon\\) is by definition unpredictable. The best prediction is \\(\\hat{f}(x) = f(x)\\), giving \\(\\mathbb{E}[(y - f(x))^2] = \\mathbb{E}[\\epsilon^2] = \\sigma^2\\). No increase in model complexity can reduce this term.'
                },
                {
                    question: 'A linear regression model on a highly nonlinear dataset has training error 8.2 and test error 8.5. A degree-20 polynomial has training error 0.01 and test error 25.3. Diagnose each model in terms of bias and variance.',
                    hint: 'Compare the gap between training and test error for each model. Which model is underfitting? Which is overfitting?',
                    solution: 'The linear model has high bias (training error 8.2 is large, indicating the model cannot capture the pattern) and low variance (the 0.3 gap between train and test is small). It is underfitting. The degree-20 polynomial has low bias (training error 0.01 shows it captures the data perfectly) but extremely high variance (the gap of 25.29 between train and test shows it is fitting noise). It is overfitting.'
                },
                {
                    question: 'Suppose you train an ensemble of 100 neural networks on different bootstrap samples of the same dataset. For a given input \\(x_0\\), the 100 predictions are \\(\\hat{f}_1(x_0), \\ldots, \\hat{f}_{100}(x_0)\\). How would you estimate the bias and variance components from these predictions?',
                    hint: 'The average of the 100 predictions approximates \\(\\mathbb{E}_D[\\hat{f}(x_0)]\\). What do the spread of predictions tell you?',
                    solution: 'Estimate \\(\\bar{f}(x_0) = \\frac{1}{100}\\sum_{i=1}^{100}\\hat{f}_i(x_0)\\). The bias estimate is \\(\\bar{f}(x_0) - f(x_0)\\) (if \\(f(x_0)\\) is known). The variance estimate is \\(\\frac{1}{100}\\sum_{i=1}^{100}(\\hat{f}_i(x_0) - \\bar{f}(x_0))^2\\). If predictions cluster tightly around a wrong value, bias is high and variance is low. If predictions are scattered, variance is high.'
                }
            ]
        },

        // ===== SECTION 2: L1 & L2 Regularization =====
        {
            id: 'l1-l2-regularization',
            title: 'L1 & L2 Regularization',
            content: `
                <h2>Constraining Weights to Improve Generalization</h2>

                <div class="env-block intuition">
                    <div class="env-title">The Core Idea</div>
                    <div class="env-body">
                        <p>Overfitting often manifests as large weight values: the model contorts itself into extreme configurations to fit every data point, including noise. Regularization combats this by adding a penalty on weight magnitudes to the loss function, effectively telling the model: "fit the data, but keep your weights small."</p>
                    </div>
                </div>

                <h3>The Regularized Objective</h3>

                <div class="env-block definition">
                    <div class="env-title">L2 Regularization (Ridge / Weight Decay)</div>
                    <div class="env-body">
                        <p>The L2-regularized objective penalizes the squared Euclidean norm of the weights:</p>
                        \\[
                        \\tilde{J}(\\mathbf{w}) = J(\\mathbf{w}) + \\frac{\\lambda}{2} \\|\\mathbf{w}\\|_2^2
                        = J(\\mathbf{w}) + \\frac{\\lambda}{2} \\sum_i w_i^2
                        \\]
                        <p>The hyperparameter \\(\\lambda &gt; 0\\) controls the regularization strength. The gradient update becomes:</p>
                        \\[
                        \\mathbf{w} \\leftarrow \\mathbf{w} - \\eta\\nabla J(\\mathbf{w}) - \\eta\\lambda\\mathbf{w}
                        = (1 - \\eta\\lambda)\\mathbf{w} - \\eta\\nabla J(\\mathbf{w})
                        \\]
                        <p>The multiplicative factor \\((1 - \\eta\\lambda)\\) shrinks weights toward zero each step, which is why L2 regularization is also called <strong>weight decay</strong>.</p>
                    </div>
                </div>

                <div class="env-block definition">
                    <div class="env-title">L1 Regularization (Lasso)</div>
                    <div class="env-body">
                        <p>The L1-regularized objective penalizes the sum of absolute values:</p>
                        \\[
                        \\tilde{J}(\\mathbf{w}) = J(\\mathbf{w}) + \\lambda \\|\\mathbf{w}\\|_1
                        = J(\\mathbf{w}) + \\lambda \\sum_i |w_i|
                        \\]
                        <p>The subgradient of \\(|w_i|\\) is \\(\\text{sign}(w_i)\\), so the update for each weight is:</p>
                        \\[
                        w_i \\leftarrow w_i - \\eta\\frac{\\partial J}{\\partial w_i} - \\eta\\lambda\\,\\text{sign}(w_i)
                        \\]
                        <p>This pushes weights by a <em>constant amount</em> toward zero, independent of how large they are. Small weights can be driven exactly to zero, producing <strong>sparse</strong> solutions.</p>
                    </div>
                </div>

                <h3>Geometric Interpretation</h3>

                <p>Think of regularization as a constrained optimization problem. Minimizing \\(J(\\mathbf{w})\\) subject to \\(\\|\\mathbf{w}\\|_p \\leq t\\) is equivalent (via the Lagrangian) to adding a penalty \\(\\lambda\\|\\mathbf{w}\\|_p\\). The constraint region in weight space has a distinctive shape:</p>

                <ul>
                    <li><strong>L2 (\\(p = 2\\)):</strong> The constraint region is a <em>circle</em> (or hypersphere). The loss contours typically meet the circle smoothly, yielding small but non-zero weights.</li>
                    <li><strong>L1 (\\(p = 1\\)):</strong> The constraint region is a <em>diamond</em> (or cross-polytope). Its corners lie on the coordinate axes. The loss contours are much more likely to hit a corner, which corresponds to some weights being exactly zero.</li>
                </ul>

                <div class="viz-placeholder" data-viz="viz-l1-l2"></div>

                <div class="env-block theorem">
                    <div class="env-title">L2 Regularization in Linear Regression (Ridge Regression)</div>
                    <div class="env-body">
                        <p>For linear regression with design matrix \\(\\mathbf{X}\\), the unregularized OLS solution is \\(\\hat{\\mathbf{w}} = (\\mathbf{X}^\\top\\mathbf{X})^{-1}\\mathbf{X}^\\top\\mathbf{y}\\). The L2-regularized (ridge) solution is:</p>
                        \\[
                        \\hat{\\mathbf{w}}_{\\text{ridge}} = (\\mathbf{X}^\\top\\mathbf{X} + \\lambda\\mathbf{I})^{-1}\\mathbf{X}^\\top\\mathbf{y}
                        \\]
                        <p>The addition of \\(\\lambda\\mathbf{I}\\) ensures the matrix is always invertible (addressing multicollinearity) and shrinks the solution toward the origin. In the eigendecomposition \\(\\mathbf{X}^\\top\\mathbf{X} = \\mathbf{Q}\\boldsymbol{\\Lambda}\\mathbf{Q}^\\top\\), ridge rescales each component by \\(\\frac{\\lambda_i}{\\lambda_i + \\lambda}\\), shrinking directions with small eigenvalues (poorly determined by data) the most.</p>
                    </div>
                </div>

                <div class="env-block remark">
                    <div class="env-title">Why L1 Produces Sparsity But L2 Does Not</div>
                    <div class="env-body">
                        <p>The L1 penalty contributes a constant magnitude \\(\\lambda\\) to the gradient, regardless of \\(|w_i|\\). When \\(|w_i|\\) is small enough that the data gradient \\(\\partial J/\\partial w_i\\) is less than \\(\\lambda\\), the regularization term dominates and drives \\(w_i\\) to exactly zero. In contrast, L2 penalty's gradient is \\(\\lambda w_i\\), which shrinks proportionally: as \\(w_i\\) approaches zero, the penalty gradient also vanishes, so weights approach but never exactly reach zero.</p>
                    </div>
                </div>

                <div class="env-block warning">
                    <div class="env-title">Bias Terms Should Not Be Regularized</div>
                    <div class="env-body">
                        <p>Regularization is applied only to the weights \\(\\mathbf{w}\\), not to the bias \\(b\\). Regularizing the bias would prevent the model from shifting predictions up or down to match the data mean, introducing unnecessary bias into the estimator.</p>
                    </div>
                </div>

                <h3>Elastic Net: The Best of Both Worlds</h3>

                <p>The <strong>elastic net</strong> combines L1 and L2 penalties:</p>
                \\[
                \\tilde{J}(\\mathbf{w}) = J(\\mathbf{w}) + \\lambda_1\\|\\mathbf{w}\\|_1 + \\frac{\\lambda_2}{2}\\|\\mathbf{w}\\|_2^2
                \\]
                <p>This retains the sparsity-inducing property of L1 while gaining the grouping effect of L2 (correlated features get similar weights rather than one being zeroed arbitrarily).</p>
            `,
            visualizations: [
                {
                    id: 'viz-l1-l2',
                    title: 'L1 vs L2 Constraint Regions with Loss Contours',
                    description: 'The diamond (L1) has corners on axes where solutions tend to land, producing zero weights. The circle (L2) is smooth, so solutions rarely have exactly zero weights. Drag the slider to change the constraint size.',
                    setup: function(body, controls) {
                        var viz = new VizEngine(body, { width: 800, height: 400, scale: 1, originX: 0, originY: 0 });
                        var ctx = viz.ctx;
                        var W = viz.width, H = viz.height;
                        var constraintSize = 1.5;
                        // Loss minimum location
                        var lossCx = 1.8, lossCy = 1.0;

                        function draw() {
                            viz.clear();
                            var halfW = W / 2;

                            // Draw both panels
                            drawPanel(0, 'L1 (Lasso)', 'diamond');
                            drawPanel(halfW, 'L2 (Ridge)', 'circle');

                            // Separator
                            ctx.strokeStyle = viz.colors.axis;
                            ctx.lineWidth = 1;
                            ctx.beginPath(); ctx.moveTo(halfW, 0); ctx.lineTo(halfW, H); ctx.stroke();
                        }

                        function drawPanel(offX, label, type) {
                            var margin = 40;
                            var pW = W / 2 - 2 * margin;
                            var pH = H - 2 * margin;
                            var cx = offX + W / 4;
                            var cy = H / 2;
                            var scale = Math.min(pW, pH) / 6;

                            function toPixel(wx, wy) {
                                return [cx + wx * scale, cy - wy * scale];
                            }

                            // Title
                            ctx.fillStyle = viz.colors.white;
                            ctx.font = 'bold 14px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText(label, cx, 22);

                            // Axes
                            ctx.strokeStyle = viz.colors.axis;
                            ctx.lineWidth = 1;
                            var axLen = 3 * scale;
                            ctx.beginPath(); ctx.moveTo(cx - axLen, cy); ctx.lineTo(cx + axLen, cy); ctx.stroke();
                            ctx.beginPath(); ctx.moveTo(cx, cy - axLen); ctx.lineTo(cx, cy + axLen); ctx.stroke();

                            // Axis labels
                            ctx.fillStyle = viz.colors.text;
                            ctx.font = '12px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText('w\u2081', cx + axLen + 10, cy + 4);
                            ctx.fillText('w\u2082', cx, cy - axLen - 8);

                            // Loss contours (ellipses centered at lossCx, lossCy)
                            var nContours = 8;
                            for (var i = 1; i <= nContours; i++) {
                                var r = i * 0.4;
                                var p = toPixel(lossCx, lossCy);
                                ctx.strokeStyle = 'rgba(248,81,73,0.2)';
                                ctx.lineWidth = 1;
                                ctx.beginPath();
                                ctx.ellipse(p[0], p[1], r * scale * 0.7, r * scale * 1.0, -Math.PI / 6, 0, Math.PI * 2);
                                ctx.stroke();
                            }

                            // Loss minimum marker
                            var lossP = toPixel(lossCx, lossCy);
                            ctx.fillStyle = viz.colors.red;
                            ctx.beginPath(); ctx.arc(lossP[0], lossP[1], 4, 0, Math.PI * 2); ctx.fill();
                            ctx.fillStyle = viz.colors.red;
                            ctx.font = '11px -apple-system,sans-serif';
                            ctx.textAlign = 'left';
                            ctx.fillText('w*', lossP[0] + 6, lossP[1] - 6);

                            // Constraint region
                            if (type === 'diamond') {
                                // L1: |w1| + |w2| <= t
                                var t = constraintSize;
                                ctx.fillStyle = 'rgba(63,185,160,0.15)';
                                ctx.strokeStyle = viz.colors.teal;
                                ctx.lineWidth = 2;
                                ctx.beginPath();
                                var p0 = toPixel(t, 0);
                                var p1 = toPixel(0, t);
                                var p2 = toPixel(-t, 0);
                                var p3 = toPixel(0, -t);
                                ctx.moveTo(p0[0], p0[1]);
                                ctx.lineTo(p1[0], p1[1]);
                                ctx.lineTo(p2[0], p2[1]);
                                ctx.lineTo(p3[0], p3[1]);
                                ctx.closePath();
                                ctx.fill();
                                ctx.stroke();

                                // Solution point: find where the contour hits the diamond
                                // Use analytical approximation: contour ellipse tangent to diamond
                                var solW1 = constraintSize;
                                var solW2 = 0;
                                // Check if hitting vertical edge instead
                                var angle = Math.atan2(lossCy, lossCx);
                                if (angle > Math.PI / 4) {
                                    solW1 = 0;
                                    solW2 = constraintSize;
                                } else {
                                    solW1 = constraintSize;
                                    solW2 = 0;
                                }

                                var solP = toPixel(solW1, solW2);
                                ctx.fillStyle = viz.colors.yellow;
                                ctx.beginPath(); ctx.arc(solP[0], solP[1], 6, 0, Math.PI * 2); ctx.fill();
                                ctx.fillStyle = '#000';
                                ctx.beginPath(); ctx.arc(solP[0], solP[1], 2, 0, Math.PI * 2); ctx.fill();

                                // Label
                                ctx.fillStyle = viz.colors.yellow;
                                ctx.font = '11px -apple-system,sans-serif';
                                ctx.textAlign = 'left';
                                if (solW2 === 0) {
                                    ctx.fillText('w\u2082 = 0 (sparse!)', solP[0] + 8, solP[1] + 4);
                                } else {
                                    ctx.fillText('w\u2081 = 0 (sparse!)', solP[0] + 8, solP[1] + 4);
                                }
                            } else {
                                // L2: w1^2 + w2^2 <= t^2
                                var t = constraintSize;
                                ctx.fillStyle = 'rgba(88,166,255,0.12)';
                                ctx.strokeStyle = viz.colors.blue;
                                ctx.lineWidth = 2;
                                var cp = toPixel(0, 0);
                                ctx.beginPath();
                                ctx.arc(cp[0], cp[1], t * scale, 0, Math.PI * 2);
                                ctx.fill();
                                ctx.stroke();

                                // Solution: project loss minimum onto circle
                                var dist = Math.sqrt(lossCx * lossCx + lossCy * lossCy);
                                var solW1, solW2;
                                if (dist <= t) {
                                    solW1 = lossCx; solW2 = lossCy;
                                } else {
                                    // Approximate: direction from origin to lossCx,lossCy, on circle boundary
                                    // Adjusted for elliptical contours
                                    var ang = Math.atan2(lossCy * 0.7, lossCx * 1.0);
                                    solW1 = t * Math.cos(ang);
                                    solW2 = t * Math.sin(ang);
                                }

                                var solP = toPixel(solW1, solW2);
                                ctx.fillStyle = viz.colors.yellow;
                                ctx.beginPath(); ctx.arc(solP[0], solP[1], 6, 0, Math.PI * 2); ctx.fill();
                                ctx.fillStyle = '#000';
                                ctx.beginPath(); ctx.arc(solP[0], solP[1], 2, 0, Math.PI * 2); ctx.fill();

                                ctx.fillStyle = viz.colors.yellow;
                                ctx.font = '11px -apple-system,sans-serif';
                                ctx.textAlign = 'left';
                                ctx.fillText('Both w\u2081, w\u2082 \u2260 0', solP[0] + 8, solP[1] - 8);
                            }
                        }

                        draw();

                        VizEngine.createSlider(controls, 'Constraint t', 0.3, 2.5, 1.5, 0.1, function(val) {
                            constraintSize = val;
                            draw();
                        });
                    }
                }
            ],
            exercises: [
                {
                    question: 'Derive the gradient of the L2-regularized loss \\(\\tilde{J}(\\mathbf{w}) = \\frac{1}{2}\\|\\mathbf{y} - \\mathbf{Xw}\\|^2 + \\frac{\\lambda}{2}\\|\\mathbf{w}\\|^2\\) for linear regression, and show that the solution satisfies \\((\\mathbf{X}^\\top\\mathbf{X} + \\lambda\\mathbf{I})\\mathbf{w} = \\mathbf{X}^\\top\\mathbf{y}\\).',
                    hint: 'Compute \\(\\nabla_{\\mathbf{w}}\\tilde{J}\\) by differentiating each term. Set the gradient to zero.',
                    solution: 'We have \\(\\nabla_{\\mathbf{w}}\\tilde{J} = -\\mathbf{X}^\\top(\\mathbf{y} - \\mathbf{Xw}) + \\lambda\\mathbf{w} = -\\mathbf{X}^\\top\\mathbf{y} + \\mathbf{X}^\\top\\mathbf{Xw} + \\lambda\\mathbf{w}\\). Setting this to zero: \\((\\mathbf{X}^\\top\\mathbf{X} + \\lambda\\mathbf{I})\\mathbf{w} = \\mathbf{X}^\\top\\mathbf{y}\\). Since \\(\\mathbf{X}^\\top\\mathbf{X} + \\lambda\\mathbf{I}\\) is always positive definite for \\(\\lambda > 0\\), it is invertible, giving \\(\\mathbf{w} = (\\mathbf{X}^\\top\\mathbf{X} + \\lambda\\mathbf{I})^{-1}\\mathbf{X}^\\top\\mathbf{y}\\).'
                },
                {
                    question: 'Explain geometrically why L1 regularization tends to produce sparse solutions while L2 does not. Use the concepts of constraint regions and loss contours in your explanation.',
                    hint: 'Think about the shapes: a diamond has corners on the axes, while a circle is smooth everywhere.',
                    solution: 'The L1 constraint region \\(\\|\\mathbf{w}\\|_1 \\leq t\\) is a diamond whose corners lie on the coordinate axes (where some weights are exactly zero). The elliptical loss contours, expanding from the unconstrained minimum, will generically first touch the diamond at one of these corners rather than along an edge. At a corner, at least one weight is exactly zero. In contrast, the L2 constraint region is a circle with no corners. The loss contours meet it tangentially at a smooth point, where all weight components are generically non-zero.'
                },
                {
                    question: 'Suppose the eigenvalues of \\(\\mathbf{X}^\\top\\mathbf{X}\\) are \\(\\{100, 1, 0.01\\}\\). Compute the effective shrinkage factor \\(\\frac{\\lambda_i}{\\lambda_i + \\lambda}\\) for each eigenvalue when \\(\\lambda = 1\\). Which direction is most affected by ridge regularization?',
                    hint: 'Plug each eigenvalue into the formula. The direction with the smallest eigenvalue will be shrunk the most.',
                    solution: 'Shrinkage factors: \\(\\frac{100}{101} \\approx 0.99\\), \\(\\frac{1}{2} = 0.50\\), \\(\\frac{0.01}{1.01} \\approx 0.0099\\). The direction corresponding to eigenvalue 0.01 is shrunk by 99%, nearly eliminated. This direction is poorly determined by the data (small variance), so ridge regularization correctly suppresses it the most. The well-determined direction (eigenvalue 100) is barely affected.'
                }
            ]
        },

        // ===== SECTION 3: Dropout =====
        {
            id: 'dropout',
            title: 'Dropout',
            content: `
                <h2>Training with Random Subnetworks</h2>

                <div class="env-block intuition">
                    <div class="env-title">The Ensemble Interpretation</div>
                    <div class="env-body">
                        <p>Imagine training not one large network, but an exponential number of smaller networks that share parameters. Each "sub-network" sees different neurons active, forcing every neuron to be useful on its own rather than relying on specific co-adaptations with other neurons. At test time, you effectively average all these sub-networks together. This is the intuition behind <strong>dropout</strong> (Srivastava et al., 2014).</p>
                    </div>
                </div>

                <h3>The Dropout Algorithm</h3>

                <div class="env-block definition">
                    <div class="env-title">Dropout During Training</div>
                    <div class="env-body">
                        <p>For each training example, each hidden unit is independently set to zero with probability \\(p\\) (the <em>dropout rate</em>). Equivalently, each unit is kept with probability \\(q = 1 - p\\). Formally, let \\(\\mathbf{m}\\) be a binary mask where each element \\(m_j \\sim \\text{Bernoulli}(q)\\). The forward pass becomes:</p>
                        \\[
                        \\tilde{\\mathbf{h}} = \\mathbf{m} \\odot \\mathbf{h}
                        \\]
                        <p>where \\(\\odot\\) denotes element-wise multiplication and \\(\\mathbf{h}\\) is the original activation vector. This mask is resampled for each training example (or mini-batch).</p>
                    </div>
                </div>

                <div class="env-block definition">
                    <div class="env-title">Dropout During Inference (Inverted Dropout)</div>
                    <div class="env-body">
                        <p>At test time, all neurons are active (no masking). To compensate for the fact that there are now more active units than during training, we scale the activations. In the popular <strong>inverted dropout</strong> scheme, the scaling is applied during training:</p>
                        \\[
                        \\tilde{\\mathbf{h}} = \\frac{1}{q}(\\mathbf{m} \\odot \\mathbf{h})
                        \\]
                        <p>This ensures \\(\\mathbb{E}[\\tilde{\\mathbf{h}}] = \\mathbf{h}\\), so no modification is needed at test time. The alternative (scaling by \\(q\\) at test time) is mathematically equivalent but less convenient.</p>
                    </div>
                </div>

                <div class="viz-placeholder" data-viz="viz-dropout"></div>

                <h3>Why Dropout Works</h3>

                <p>Several complementary perspectives explain dropout's effectiveness:</p>

                <div class="env-block remark">
                    <div class="env-title">Perspective 1: Implicit Ensemble</div>
                    <div class="env-body">
                        <p>A network with \\(n\\) units has \\(2^n\\) possible sub-networks (each unit is either on or off). Dropout trains a different sub-network each iteration, and the full network at test time approximates the geometric average of all \\(2^n\\) sub-networks. This is a powerful ensemble with shared parameters.</p>
                    </div>
                </div>

                <div class="env-block remark">
                    <div class="env-title">Perspective 2: Breaking Co-adaptation</div>
                    <div class="env-body">
                        <p>Without dropout, hidden units can develop complex co-dependencies: unit A relies on unit B to correct its errors, creating a fragile representation. Dropout forces each unit to learn features that are individually useful, because it cannot predict which other units will be present. This produces more robust, distributed representations.</p>
                    </div>
                </div>

                <div class="env-block remark">
                    <div class="env-title">Perspective 3: Noise Injection as Regularization</div>
                    <div class="env-body">
                        <p>Dropout can be viewed as adding multiplicative Bernoulli noise to the hidden representations. This noise injection prevents the network from relying too heavily on any single feature, functioning similarly to data augmentation applied in the representation space.</p>
                    </div>
                </div>

                <h3>Practical Guidelines</h3>
                <ul>
                    <li><strong>Common dropout rates:</strong> \\(p = 0.5\\) for hidden layers, \\(p = 0.2\\) for the input layer.</li>
                    <li><strong>Where to apply:</strong> Typically after fully-connected layers. Less commonly used after convolutional layers (spatial dropout or DropBlock is preferred there).</li>
                    <li><strong>Effect on training:</strong> Dropout slows convergence because each update uses a thinner network. Train for more epochs or use higher learning rates.</li>
                    <li><strong>Batch normalization interaction:</strong> Dropout and batch norm can conflict because dropout changes the variance that batch norm calibrates to. In modern architectures, batch norm often replaces dropout.</li>
                </ul>

                <div class="env-block warning">
                    <div class="env-title">Dropout Must Be Disabled at Test Time</div>
                    <div class="env-body">
                        <p>A common implementation bug is to forget to switch dropout off during evaluation. In frameworks like PyTorch, call <code>model.eval()</code> before inference and <code>model.train()</code> before training. Leaving dropout active at test time introduces random noise into predictions and degrades performance.</p>
                    </div>
                </div>
            `,
            visualizations: [
                {
                    id: 'viz-dropout',
                    title: 'Dropout: Random Sub-network Sampling',
                    description: 'Each forward pass uses a different random mask, creating a different sub-network. Press "New Mask" to resample, or watch the animation cycle through different masks.',
                    setup: function(body, controls) {
                        var viz = new VizEngine(body, { width: 700, height: 400, scale: 1, originX: 0, originY: 0 });
                        var ctx = viz.ctx;
                        var W = viz.width, H = viz.height;
                        var dropRate = 0.5;
                        var animating = true;
                        var lastMaskTime = 0;
                        var maskInterval = 1200;

                        // Network architecture
                        var layers = [4, 6, 6, 3];
                        var layerNames = ['Input', 'Hidden 1', 'Hidden 2', 'Output'];
                        var mask = [];

                        function generateMask() {
                            mask = [];
                            for (var l = 0; l < layers.length; l++) {
                                var layerMask = [];
                                for (var n = 0; n < layers[l]; n++) {
                                    // Don't drop input or output neurons
                                    if (l === 0 || l === layers.length - 1) {
                                        layerMask.push(1);
                                    } else {
                                        layerMask.push(Math.random() > dropRate ? 1 : 0);
                                    }
                                }
                                // Ensure at least one neuron per hidden layer is active
                                if (l > 0 && l < layers.length - 1) {
                                    var anyActive = layerMask.some(function(m) { return m === 1; });
                                    if (!anyActive) {
                                        layerMask[Math.floor(Math.random() * layerMask.length)] = 1;
                                    }
                                }
                                mask.push(layerMask);
                            }
                        }

                        generateMask();

                        function getNeuronPos(layerIdx, neuronIdx) {
                            var nLayers = layers.length;
                            var layerSpacing = (W - 160) / (nLayers - 1);
                            var x = 80 + layerIdx * layerSpacing;
                            var nNeurons = layers[layerIdx];
                            var neuronSpacing = Math.min(50, (H - 100) / (nNeurons + 1));
                            var startY = H / 2 - (nNeurons - 1) * neuronSpacing / 2;
                            var y = startY + neuronIdx * neuronSpacing;
                            return [x, y];
                        }

                        function drawNetwork() {
                            viz.clear();

                            // Count active neurons
                            var totalHidden = 0, activeHidden = 0;
                            for (var l = 1; l < layers.length - 1; l++) {
                                for (var n = 0; n < layers[l]; n++) {
                                    totalHidden++;
                                    if (mask[l][n]) activeHidden++;
                                }
                            }

                            // Title info
                            ctx.fillStyle = viz.colors.white;
                            ctx.font = 'bold 13px -apple-system,sans-serif';
                            ctx.textAlign = 'left';
                            ctx.fillText('Active: ' + activeHidden + '/' + totalHidden + ' hidden units', 15, 22);
                            ctx.fillStyle = viz.colors.text;
                            ctx.font = '11px -apple-system,sans-serif';
                            ctx.fillText('Drop rate p = ' + dropRate.toFixed(2) + ', keep rate q = ' + (1 - dropRate).toFixed(2), 15, 40);

                            // Draw connections
                            for (var l = 0; l < layers.length - 1; l++) {
                                for (var i = 0; i < layers[l]; i++) {
                                    for (var j = 0; j < layers[l + 1]; j++) {
                                        var active = mask[l][i] && mask[l + 1][j];
                                        var p1 = getNeuronPos(l, i);
                                        var p2 = getNeuronPos(l + 1, j);
                                        ctx.strokeStyle = active ? 'rgba(88,166,255,0.3)' : 'rgba(50,50,80,0.1)';
                                        ctx.lineWidth = active ? 1.5 : 0.5;
                                        ctx.beginPath();
                                        ctx.moveTo(p1[0] + 14, p1[1]);
                                        ctx.lineTo(p2[0] - 14, p2[1]);
                                        ctx.stroke();
                                    }
                                }
                            }

                            // Draw neurons
                            for (var l = 0; l < layers.length; l++) {
                                for (var n = 0; n < layers[l]; n++) {
                                    var pos = getNeuronPos(l, n);
                                    var active = mask[l][n];
                                    var radius = 13;

                                    if (active) {
                                        // Active neuron
                                        var color = l === 0 ? viz.colors.teal :
                                                    l === layers.length - 1 ? viz.colors.orange : viz.colors.blue;
                                        ctx.fillStyle = color + '33';
                                        ctx.beginPath(); ctx.arc(pos[0], pos[1], radius + 3, 0, Math.PI * 2); ctx.fill();
                                        ctx.fillStyle = color;
                                        ctx.beginPath(); ctx.arc(pos[0], pos[1], radius, 0, Math.PI * 2); ctx.fill();
                                        // Highlight
                                        ctx.fillStyle = 'rgba(255,255,255,0.2)';
                                        ctx.beginPath(); ctx.arc(pos[0] - 3, pos[1] - 3, 4, 0, Math.PI * 2); ctx.fill();
                                    } else {
                                        // Dropped neuron
                                        ctx.strokeStyle = 'rgba(248,81,73,0.4)';
                                        ctx.lineWidth = 1.5;
                                        ctx.setLineDash([3, 3]);
                                        ctx.beginPath(); ctx.arc(pos[0], pos[1], radius, 0, Math.PI * 2); ctx.stroke();
                                        ctx.setLineDash([]);
                                        // X mark
                                        ctx.strokeStyle = viz.colors.red;
                                        ctx.lineWidth = 2;
                                        ctx.beginPath();
                                        ctx.moveTo(pos[0] - 5, pos[1] - 5);
                                        ctx.lineTo(pos[0] + 5, pos[1] + 5);
                                        ctx.moveTo(pos[0] + 5, pos[1] - 5);
                                        ctx.lineTo(pos[0] - 5, pos[1] + 5);
                                        ctx.stroke();
                                    }
                                }
                            }

                            // Layer labels
                            ctx.textAlign = 'center';
                            for (var l = 0; l < layers.length; l++) {
                                var pos = getNeuronPos(l, 0);
                                ctx.fillStyle = viz.colors.text;
                                ctx.font = '11px -apple-system,sans-serif';
                                ctx.fillText(layerNames[l], pos[0], H - 15);
                            }

                            // Scaling note
                            ctx.fillStyle = viz.colors.teal;
                            ctx.font = '11px -apple-system,sans-serif';
                            ctx.textAlign = 'right';
                            ctx.fillText('Inverted dropout: active outputs scaled by 1/q = ' + (1 / (1 - dropRate)).toFixed(2), W - 15, 22);
                        }

                        drawNetwork();

                        viz.animate(function(t) {
                            if (animating && t - lastMaskTime > maskInterval) {
                                generateMask();
                                lastMaskTime = t;
                            }
                            drawNetwork();
                        });

                        VizEngine.createSlider(controls, 'Drop rate p', 0, 0.8, 0.5, 0.05, function(val) {
                            dropRate = val;
                            generateMask();
                        });

                        VizEngine.createButton(controls, 'New Mask', function() {
                            generateMask();
                        });

                        VizEngine.createButton(controls, animating ? 'Pause' : 'Play', function() {
                            animating = !animating;
                            this.textContent = animating ? 'Pause' : 'Play';
                        });

                        return { stopAnimation: function() { viz.stopAnimation(); } };
                    }
                }
            ],
            exercises: [
                {
                    question: 'A hidden layer has 1024 units with dropout rate \\(p = 0.5\\). (a) How many possible sub-networks can be formed by this layer alone? (b) During a single forward pass, what is the expected number of active units? (c) With inverted dropout, what scaling factor is applied?',
                    hint: 'Each unit is independently either on or off. The keep probability is \\(q = 1 - p\\).',
                    solution: '(a) Each of the 1024 units can be on or off, giving \\(2^{1024}\\) possible masks (an astronomically large number). (b) Each unit is kept with probability \\(q = 0.5\\), so the expected number of active units is \\(1024 \\times 0.5 = 512\\). (c) With inverted dropout, active outputs are scaled by \\(1/q = 1/0.5 = 2\\), ensuring the expected value of each unit\'s output remains unchanged.'
                },
                {
                    question: 'Explain why dropout is equivalent to training an ensemble of \\(2^n\\) sub-networks with shared parameters, and why the full network at test time approximates the ensemble prediction.',
                    hint: 'Each mask defines a sub-network. Training with a particular mask updates the weights of that sub-network. What does the test-time output represent in terms of these sub-networks?',
                    solution: 'With \\(n\\) units, there are \\(2^n\\) possible binary masks, each defining a sub-network. Each training step samples one mask and updates only the corresponding sub-network\'s weights (though weights are shared across sub-networks). At test time, using all units with inverted-dropout scaling computes a weighted average of all sub-network outputs. For linear models this is exact; for nonlinear networks it approximates the geometric mean of the sub-network distributions, which tends to be a good approximation because dropout masks are roughly balanced.'
                },
                {
                    question: 'You observe that your model trains slowly with dropout \\(p = 0.5\\). One colleague suggests increasing the learning rate; another suggests reducing the dropout rate. Analyze both suggestions.',
                    hint: 'Think about how dropout affects the effective gradient signal. Each update only trains a fraction of the weights.',
                    solution: 'Increasing the learning rate compensates for the fact that dropout reduces the effective gradient magnitude: with half the units dropped, the gradient signal is noisier and thinner. A moderate increase (roughly 2x) is often appropriate, though too large a rate causes instability. Reducing the dropout rate (say from 0.5 to 0.2) reduces noise but also reduces regularization; the model trains faster but may overfit more. The best approach depends on context: if the model is overfitting (large train-test gap), keep p = 0.5 and increase the learning rate. If the model is well-regularized by other means (batch norm, data augmentation), reducing p is reasonable.'
                }
            ]
        },

        // ===== SECTION 4: Data Augmentation & Early Stopping =====
        {
            id: 'data-augmentation-early-stopping',
            title: 'Data Augmentation & Early Stopping',
            content: `
                <h2>Leveraging Data and Timing to Prevent Overfitting</h2>

                <h3>Data Augmentation</h3>

                <div class="env-block intuition">
                    <div class="env-title">The Best Regularizer Is More Data</div>
                    <div class="env-body">
                        <p>More training data directly reduces variance. When real data is expensive, we can synthetically expand the dataset by applying label-preserving transformations. A cat rotated 15 degrees is still a cat; a sentence with a synonym substitution conveys the same meaning. Data augmentation encodes our knowledge of invariances into the training process.</p>
                    </div>
                </div>

                <div class="env-block definition">
                    <div class="env-title">Data Augmentation</div>
                    <div class="env-body">
                        <p>Data augmentation generates new training examples by applying transformations \\(T\\) to existing examples \\((x, y)\\) such that the label \\(y\\) is preserved (or appropriately modified):</p>
                        \\[
                        \\mathcal{D}_{\\text{aug}} = \\{(T(x), y) : (x, y) \\in \\mathcal{D},\\; T \\in \\mathcal{T}\\}
                        \\]
                        <p>The transformation family \\(\\mathcal{T}\\) should encode known invariances of the task.</p>
                    </div>
                </div>

                <h4>Common Augmentations for Images</h4>
                <ul>
                    <li><strong>Geometric:</strong> Random crops, horizontal flips, rotations, scaling, shearing, perspective warping</li>
                    <li><strong>Photometric:</strong> Color jittering, brightness/contrast changes, Gaussian blur, noise injection</li>
                    <li><strong>Erasing:</strong> Random erasing (Cutout) removes rectangular patches, forcing the model to use multiple cues</li>
                    <li><strong>Mixing:</strong> Mixup, CutMix blend multiple images and their labels (covered in Section 5)</li>
                </ul>

                <h4>Augmentations for Other Domains</h4>
                <ul>
                    <li><strong>Text:</strong> Synonym replacement, random insertion/deletion, back-translation, EDA (Easy Data Augmentation)</li>
                    <li><strong>Audio:</strong> Time stretching, pitch shifting, noise injection, SpecAugment (masking time/frequency bands)</li>
                    <li><strong>Tabular:</strong> SMOTE for class imbalance, noise injection, feature-space interpolation</li>
                </ul>

                <div class="env-block warning">
                    <div class="env-title">Augmentation Must Preserve Label Validity</div>
                    <div class="env-body">
                        <p>Not all transformations are label-preserving. Vertically flipping a digit '6' produces '9'; horizontal flipping of text renders it unreadable. Always consider the specific task when choosing augmentations. Aggressive augmentation that breaks the label correspondence will <em>hurt</em> performance.</p>
                    </div>
                </div>

                <h3>Early Stopping</h3>

                <div class="env-block definition">
                    <div class="env-title">Early Stopping</div>
                    <div class="env-body">
                        <p>Early stopping monitors a validation metric (typically validation loss) during training and stops when performance on the validation set begins to degrade. The model parameters from the epoch with the best validation performance are retained:</p>
                        \\[
                        \\hat{\\mathbf{w}} = \\mathbf{w}^{(t^*)} \\quad \\text{where} \\quad t^* = \\arg\\min_t \\; L_{\\text{val}}(\\mathbf{w}^{(t)})
                        \\]
                        <p>A <strong>patience</strong> parameter \\(k\\) specifies how many epochs of no improvement to tolerate before stopping. This prevents stopping too early due to temporary fluctuations.</p>
                    </div>
                </div>

                <div class="viz-placeholder" data-viz="viz-early-stopping"></div>

                <div class="env-block theorem">
                    <div class="env-title">Early Stopping as L2 Regularization</div>
                    <div class="env-body">
                        <p>For a quadratic loss surface with learning rate \\(\\eta\\) and eigenvalue \\(\\lambda_i\\) of the Hessian, gradient descent initialized at the origin satisfies after \\(t\\) steps:</p>
                        \\[
                        w_i^{(t)} = (1 - (1 - \\eta\\lambda_i)^t) \\, w_i^*
                        \\]
                        <p>where \\(w_i^*\\) is the optimal weight. For small \\(t\\), this is approximately \\(w_i^{(t)} \\approx \\eta t \\lambda_i w_i^*\\), which has the same form as the ridge solution with \\(\\alpha = 1/(\\eta t)\\). Thus <strong>early stopping is approximately equivalent to L2 regularization</strong>, with the number of training steps playing the role of the inverse regularization strength.</p>
                    </div>
                </div>

                <div class="env-block remark">
                    <div class="env-title">Implementation Details</div>
                    <div class="env-body">
                        <p>In practice, early stopping requires:</p>
                        <ul>
                            <li>A held-out validation set (typically 10-20% of training data)</li>
                            <li>Checkpointing the best model weights periodically</li>
                            <li>A patience hyperparameter (common values: 5-20 epochs)</li>
                            <li>Monitoring validation loss, not training loss</li>
                        </ul>
                        <p>Early stopping is appealing because it is computationally free (you save time by training less) and introduces only one hyperparameter (patience). It is nearly universally used in practice.</p>
                    </div>
                </div>

                <div class="env-block example">
                    <div class="env-title">Example: Early Stopping in PyTorch</div>
                    <div class="env-body">
                        <p>A simplified early stopping loop:</p>
<pre><code>best_val_loss = float('inf')
patience_counter = 0
for epoch in range(max_epochs):
    train_one_epoch(model, train_loader)
    val_loss = evaluate(model, val_loader)
    if val_loss &lt; best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best.pt')
    else:
        patience_counter += 1
        if patience_counter &gt;= patience:
            break
model.load_state_dict(torch.load('best.pt'))</code></pre>
                    </div>
                </div>
            `,
            visualizations: [
                {
                    id: 'viz-early-stopping',
                    title: 'Early Stopping: When to Stop Training',
                    description: 'Training loss always decreases, but validation loss eventually rises as the model overfits. Adjust patience to see how early stopping selects the optimal epoch.',
                    setup: function(body, controls) {
                        var viz = new VizEngine(body, { width: 700, height: 400, scale: 1, originX: 0, originY: 0 });
                        var ctx = viz.ctx;
                        var W = viz.width, H = viz.height;
                        var patience = 8;
                        var noiseLevel = 0.3;
                        var seed = 17;

                        function seededRandom(s) {
                            var x = Math.sin(s) * 10000;
                            return x - Math.floor(x);
                        }

                        var totalEpochs = 100;

                        function generateCurves() {
                            var trainLoss = [], valLoss = [];
                            for (var e = 0; e < totalEpochs; e++) {
                                // Training loss: monotonically decreasing with noise
                                var tl = 2.5 * Math.exp(-0.04 * e) + 0.1 + (seededRandom(seed + e * 3) - 0.5) * 0.05;
                                trainLoss.push(Math.max(0.05, tl));

                                // Validation loss: U-shaped
                                var vl = 2.5 * Math.exp(-0.035 * e) + 0.3 + 0.015 * Math.max(0, e - 25) + (seededRandom(seed + e * 7 + 100) - 0.5) * noiseLevel;
                                valLoss.push(Math.max(0.1, vl));
                            }
                            return { train: trainLoss, val: valLoss };
                        }

                        var curves = generateCurves();

                        function findEarlyStop(valLoss, pat) {
                            var bestEpoch = 0;
                            var bestVal = valLoss[0];
                            var counter = 0;
                            for (var e = 1; e < valLoss.length; e++) {
                                if (valLoss[e] < bestVal) {
                                    bestVal = valLoss[e];
                                    bestEpoch = e;
                                    counter = 0;
                                } else {
                                    counter++;
                                    if (counter >= pat) {
                                        return { stopEpoch: e, bestEpoch: bestEpoch, bestVal: bestVal };
                                    }
                                }
                            }
                            return { stopEpoch: valLoss.length - 1, bestEpoch: bestEpoch, bestVal: bestVal };
                        }

                        function draw() {
                            viz.clear();

                            var margin = { top: 50, bottom: 50, left: 60, right: 30 };
                            var plotW = W - margin.left - margin.right;
                            var plotH = H - margin.top - margin.bottom;

                            // Y range
                            var maxLoss = 0;
                            for (var i = 0; i < totalEpochs; i++) {
                                maxLoss = Math.max(maxLoss, curves.train[i], curves.val[i]);
                            }
                            maxLoss = Math.ceil(maxLoss * 2) / 2;
                            var minLoss = 0;

                            function toPixel(epoch, loss) {
                                var px = margin.left + (epoch / (totalEpochs - 1)) * plotW;
                                var py = margin.top + (1 - (loss - minLoss) / (maxLoss - minLoss)) * plotH;
                                return [px, py];
                            }

                            // Grid
                            ctx.strokeStyle = viz.colors.grid;
                            ctx.lineWidth = 0.5;
                            for (var e = 0; e <= totalEpochs; e += 10) {
                                var p = toPixel(e, 0);
                                ctx.beginPath(); ctx.moveTo(p[0], margin.top); ctx.lineTo(p[0], margin.top + plotH); ctx.stroke();
                            }
                            for (var l = 0; l <= maxLoss; l += 0.5) {
                                var p = toPixel(0, l);
                                ctx.beginPath(); ctx.moveTo(margin.left, p[1]); ctx.lineTo(margin.left + plotW, p[1]); ctx.stroke();
                            }

                            // Axes
                            ctx.strokeStyle = viz.colors.axis;
                            ctx.lineWidth = 1.5;
                            ctx.beginPath(); ctx.moveTo(margin.left, margin.top + plotH); ctx.lineTo(margin.left + plotW, margin.top + plotH); ctx.stroke();
                            ctx.beginPath(); ctx.moveTo(margin.left, margin.top); ctx.lineTo(margin.left, margin.top + plotH); ctx.stroke();

                            // Axis labels
                            ctx.fillStyle = viz.colors.text;
                            ctx.font = '11px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            for (var e = 0; e <= totalEpochs; e += 20) {
                                var p = toPixel(e, 0);
                                ctx.fillText(e, p[0], margin.top + plotH + 16);
                            }
                            ctx.fillText('Epoch', margin.left + plotW / 2, H - 10);

                            ctx.save();
                            ctx.translate(15, margin.top + plotH / 2);
                            ctx.rotate(-Math.PI / 2);
                            ctx.textAlign = 'center';
                            ctx.fillText('Loss', 0, 0);
                            ctx.restore();

                            // Y tick labels
                            ctx.textAlign = 'right';
                            for (var l = 0; l <= maxLoss; l += 0.5) {
                                var p = toPixel(0, l);
                                ctx.fillText(l.toFixed(1), margin.left - 8, p[1] + 3);
                            }

                            // Early stopping analysis
                            var es = findEarlyStop(curves.val, patience);

                            // Overfitting region
                            if (es.bestEpoch < totalEpochs - 1) {
                                var p1 = toPixel(es.bestEpoch, maxLoss);
                                var p2 = toPixel(totalEpochs - 1, minLoss);
                                ctx.fillStyle = 'rgba(248,81,73,0.06)';
                                ctx.fillRect(p1[0], margin.top, p2[0] - p1[0], plotH);

                                ctx.fillStyle = viz.colors.red;
                                ctx.font = '11px -apple-system,sans-serif';
                                ctx.textAlign = 'center';
                                ctx.globalAlpha = 0.6;
                                ctx.fillText('Overfitting zone', (p1[0] + p2[0]) / 2, margin.top + 15);
                                ctx.globalAlpha = 1;
                            }

                            // Training loss curve
                            ctx.strokeStyle = viz.colors.blue;
                            ctx.lineWidth = 2;
                            ctx.beginPath();
                            for (var i = 0; i < totalEpochs; i++) {
                                var p = toPixel(i, curves.train[i]);
                                if (i === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]);
                            }
                            ctx.stroke();

                            // Validation loss curve
                            ctx.strokeStyle = viz.colors.red;
                            ctx.lineWidth = 2;
                            ctx.beginPath();
                            for (var i = 0; i < totalEpochs; i++) {
                                var p = toPixel(i, curves.val[i]);
                                if (i === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]);
                            }
                            ctx.stroke();

                            // Best epoch marker
                            var bestP = toPixel(es.bestEpoch, curves.val[es.bestEpoch]);
                            ctx.strokeStyle = viz.colors.green;
                            ctx.lineWidth = 1.5;
                            ctx.setLineDash([5, 3]);
                            ctx.beginPath(); ctx.moveTo(bestP[0], margin.top); ctx.lineTo(bestP[0], margin.top + plotH); ctx.stroke();
                            ctx.setLineDash([]);

                            // Best epoch dot
                            ctx.fillStyle = viz.colors.green;
                            ctx.beginPath(); ctx.arc(bestP[0], bestP[1], 6, 0, Math.PI * 2); ctx.fill();
                            ctx.fillStyle = '#000';
                            ctx.beginPath(); ctx.arc(bestP[0], bestP[1], 2, 0, Math.PI * 2); ctx.fill();

                            // Stop epoch marker
                            var stopP = toPixel(es.stopEpoch, curves.val[Math.min(es.stopEpoch, totalEpochs - 1)]);
                            ctx.strokeStyle = viz.colors.yellow;
                            ctx.lineWidth = 1.5;
                            ctx.setLineDash([3, 3]);
                            ctx.beginPath(); ctx.moveTo(stopP[0], margin.top); ctx.lineTo(stopP[0], margin.top + plotH); ctx.stroke();
                            ctx.setLineDash([]);

                            // Legend and info
                            ctx.font = '11px -apple-system,sans-serif';
                            ctx.textAlign = 'left';
                            var legX = margin.left + 10, legY = margin.top + 10;

                            ctx.fillStyle = viz.colors.blue;
                            ctx.fillRect(legX, legY - 4, 16, 3);
                            ctx.fillStyle = viz.colors.text;
                            ctx.fillText('Training loss', legX + 22, legY);

                            ctx.fillStyle = viz.colors.red;
                            ctx.fillRect(legX, legY + 14, 16, 3);
                            ctx.fillStyle = viz.colors.text;
                            ctx.fillText('Validation loss', legX + 22, legY + 18);

                            ctx.fillStyle = viz.colors.green;
                            ctx.beginPath(); ctx.arc(legX + 8, legY + 36, 4, 0, Math.PI * 2); ctx.fill();
                            ctx.fillStyle = viz.colors.text;
                            ctx.fillText('Best epoch: ' + es.bestEpoch + ' (val loss: ' + es.bestVal.toFixed(3) + ')', legX + 22, legY + 38);

                            ctx.fillStyle = viz.colors.yellow;
                            ctx.fillRect(legX, legY + 50, 16, 3);
                            ctx.fillStyle = viz.colors.text;
                            ctx.fillText('Stop epoch: ' + es.stopEpoch + ' (patience: ' + patience + ')', legX + 22, legY + 56);
                        }

                        draw();

                        VizEngine.createSlider(controls, 'Patience', 1, 30, 8, 1, function(val) {
                            patience = Math.round(val);
                            draw();
                        });

                        VizEngine.createSlider(controls, 'Noise', 0.05, 0.8, 0.3, 0.05, function(val) {
                            noiseLevel = val;
                            curves = generateCurves();
                            draw();
                        });

                        VizEngine.createButton(controls, 'New Curves', function() {
                            seed = Math.floor(Math.random() * 10000);
                            curves = generateCurves();
                            draw();
                        });
                    }
                }
            ],
            exercises: [
                {
                    question: 'You are training an image classifier on 5,000 images. List three augmentation strategies you would apply, and for each, explain what invariance it encodes and one transformation that would NOT be label-preserving for this task.',
                    hint: 'Think about what changes in the image should not change the label (e.g., small rotations), and what changes would (e.g., large rotations for digit recognition).',
                    solution: '(1) <strong>Random horizontal flip</strong>: encodes left-right symmetry. NOT valid for recognizing letters (a flipped "b" becomes "d"). (2) <strong>Random crop + resize</strong>: encodes translation and scale invariance. NOT valid if the crop removes the object entirely (crop too aggressively). (3) <strong>Color jittering</strong>: encodes illumination invariance (the same object under different lighting). NOT valid for tasks where color is the target (e.g., classifying ripe vs. unripe fruit by color). Each augmentation should reflect a genuine invariance of the specific task.'
                },
                {
                    question: 'Explain the connection between early stopping and L2 regularization. If you train with learning rate \\(\\eta = 0.01\\) and stop at epoch \\(t = 200\\), what is the approximate equivalent L2 penalty strength \\(\\alpha\\)?',
                    hint: 'For a quadratic loss, early stopping at step \\(t\\) is equivalent to L2 regularization with \\(\\alpha \\approx 1/(\\eta t)\\).',
                    solution: 'For a quadratic loss surface, gradient descent from the origin reaches \\(w_i^{(t)} = (1 - (1 - \\eta\\lambda_i)^t)w_i^*\\). For small \\(\\eta\\lambda_i\\), this approximates \\(w_i^{(t)} \\approx \\eta t \\lambda_i w_i^*\\), which matches the ridge solution \\(w_i^{\\text{ridge}} = \\frac{\\lambda_i}{\\lambda_i + \\alpha}w_i^*\\) when \\(\\alpha \\approx 1/(\\eta t)\\). With \\(\\eta = 0.01\\) and \\(t = 200\\), the equivalent L2 strength is \\(\\alpha \\approx 1/(0.01 \\times 200) = 0.5\\). Training longer reduces the effective regularization.'
                },
                {
                    question: 'Your model has a large gap between training loss (0.12) and validation loss (0.85) at epoch 100, with the validation loss curve still rising. Explain whether you should (a) increase patience, (b) add data augmentation, or (c) reduce model capacity. Justify each recommendation.',
                    hint: 'The large gap is a clear sign of overfitting. Consider how each strategy addresses variance.',
                    solution: '(a) <strong>Do NOT increase patience</strong>: patience determines when to stop, but the validation loss is already rising. If anything, you should reduce patience or stop now. The best model was likely at an earlier epoch. (b) <strong>Add data augmentation</strong>: YES. Augmentation effectively increases dataset size and encodes invariances, directly reducing variance. This addresses the root cause (too little data relative to model capacity). (c) <strong>Reduce model capacity</strong>: Possibly. A simpler model has lower variance, which may reduce the gap. However, this also increases bias. The preferred approach is (b) first, as it reduces variance without sacrificing capacity.'
                }
            ]
        },

        // ===== SECTION 5: Practical Regularization =====
        {
            id: 'practical-regularization',
            title: 'Practical Regularization',
            content: `
                <h2>Combining Techniques and Advanced Methods</h2>

                <div class="env-block intuition">
                    <div class="env-title">Regularization as a Toolbox</div>
                    <div class="env-body">
                        <p>No single regularization technique is best in all scenarios. Modern deep learning practitioners combine several techniques, each addressing a different failure mode. The art is in choosing the right combination and tuning their strengths. This section covers advanced methods and practical guidelines for putting the pieces together.</p>
                    </div>
                </div>

                <h3>Label Smoothing</h3>

                <div class="env-block definition">
                    <div class="env-title">Label Smoothing</div>
                    <div class="env-body">
                        <p>Standard classification uses hard targets: the correct class has probability 1, all others have 0. Label smoothing (Szegedy et al., 2016) replaces the hard target \\(y_k = 1\\) with a soft target:</p>
                        \\[
                        \\tilde{y}_k = (1 - \\varepsilon) \\cdot y_k + \\frac{\\varepsilon}{K}
                        \\]
                        <p>where \\(K\\) is the number of classes and \\(\\varepsilon\\) is the smoothing parameter (typically 0.1). This prevents the model from becoming overconfident by assigning some probability mass to incorrect classes.</p>
                    </div>
                </div>

                <p>Label smoothing has a geometric effect: it encourages the penultimate-layer representations of different classes to cluster at equal distances from each other, producing better-calibrated probabilities. The cross-entropy loss with label smoothing becomes:</p>
                \\[
                L_{\\text{LS}} = (1 - \\varepsilon)\\,H(y, p) + \\varepsilon\\,H(u, p)
                \\]
                <p>where \\(H(y, p)\\) is the standard cross-entropy and \\(H(u, p)\\) is the cross-entropy with the uniform distribution. This second term acts as a KL divergence regularizer, penalizing predictions that are too peaked.</p>

                <h3>Mixup and CutMix</h3>

                <div class="env-block definition">
                    <div class="env-title">Mixup (Zhang et al., 2018)</div>
                    <div class="env-body">
                        <p>Mixup creates virtual training examples by taking convex combinations of pairs of examples and their labels:</p>
                        \\[
                        \\tilde{x} = \\lambda x_i + (1 - \\lambda) x_j, \\quad
                        \\tilde{y} = \\lambda y_i + (1 - \\lambda) y_j
                        \\]
                        <p>where \\(\\lambda \\sim \\text{Beta}(\\alpha, \\alpha)\\) for a hyperparameter \\(\\alpha\\) (typically \\(\\alpha = 0.2\\)). This encourages the model to behave linearly between training examples, reducing oscillatory behavior and improving generalization.</p>
                    </div>
                </div>

                <div class="env-block definition">
                    <div class="env-title">CutMix (Yun et al., 2019)</div>
                    <div class="env-body">
                        <p>CutMix replaces a rectangular region of one image with a patch from another, mixing the labels proportionally to the area:</p>
                        \\[
                        \\tilde{x} = \\mathbf{M} \\odot x_i + (\\mathbf{1} - \\mathbf{M}) \\odot x_j, \\quad
                        \\tilde{y} = \\lambda y_i + (1 - \\lambda) y_j
                        \\]
                        <p>where \\(\\mathbf{M}\\) is a binary mask and \\(\\lambda\\) is the proportion of the remaining area. Unlike Mixup, CutMix preserves local pixel statistics, producing more natural-looking augmented images.</p>
                    </div>
                </div>

                <h3>Combining Regularization Techniques</h3>

                <div class="viz-placeholder" data-viz="viz-reg-comparison"></div>

                <p>The following table summarizes when to use each technique:</p>

                <table style="width:100%;border-collapse:collapse;margin:16px 0;">
                    <thead>
                        <tr style="border-bottom:2px solid #30363d;">
                            <th style="text-align:left;padding:8px;color:#f0f6fc;">Technique</th>
                            <th style="text-align:left;padding:8px;color:#f0f6fc;">Mechanism</th>
                            <th style="text-align:left;padding:8px;color:#f0f6fc;">Best For</th>
                            <th style="text-align:left;padding:8px;color:#f0f6fc;">Cost</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr style="border-bottom:1px solid #21262d;">
                            <td style="padding:8px;color:#58a6ff;">L2 (Weight Decay)</td>
                            <td style="padding:8px;">Shrinks weights toward zero</td>
                            <td style="padding:8px;">Always; default for most models</td>
                            <td style="padding:8px;">Negligible</td>
                        </tr>
                        <tr style="border-bottom:1px solid #21262d;">
                            <td style="padding:8px;color:#58a6ff;">L1</td>
                            <td style="padding:8px;">Produces sparse weights</td>
                            <td style="padding:8px;">Feature selection, pruning</td>
                            <td style="padding:8px;">Negligible</td>
                        </tr>
                        <tr style="border-bottom:1px solid #21262d;">
                            <td style="padding:8px;color:#58a6ff;">Dropout</td>
                            <td style="padding:8px;">Random unit masking</td>
                            <td style="padding:8px;">FC layers; less useful with BatchNorm</td>
                            <td style="padding:8px;">Slower convergence</td>
                        </tr>
                        <tr style="border-bottom:1px solid #21262d;">
                            <td style="padding:8px;color:#58a6ff;">Data Augmentation</td>
                            <td style="padding:8px;">Expands dataset with invariances</td>
                            <td style="padding:8px;">Images, audio; moderate for text</td>
                            <td style="padding:8px;">Compute per sample</td>
                        </tr>
                        <tr style="border-bottom:1px solid #21262d;">
                            <td style="padding:8px;color:#58a6ff;">Early Stopping</td>
                            <td style="padding:8px;">Limits optimization steps</td>
                            <td style="padding:8px;">Always; free regularization</td>
                            <td style="padding:8px;">Requires val set</td>
                        </tr>
                        <tr style="border-bottom:1px solid #21262d;">
                            <td style="padding:8px;color:#58a6ff;">Label Smoothing</td>
                            <td style="padding:8px;">Softens targets</td>
                            <td style="padding:8px;">Classification; improves calibration</td>
                            <td style="padding:8px;">Negligible</td>
                        </tr>
                        <tr>
                            <td style="padding:8px;color:#58a6ff;">Mixup / CutMix</td>
                            <td style="padding:8px;">Interpolates examples</td>
                            <td style="padding:8px;">Image classification; large datasets</td>
                            <td style="padding:8px;">Moderate</td>
                        </tr>
                    </tbody>
                </table>

                <h3>A Practical Recipe</h3>

                <div class="env-block remark">
                    <div class="env-title">Modern Regularization Recipe (ResNet-style)</div>
                    <div class="env-body">
                        <p>A robust starting point for image classification:</p>
                        <ol>
                            <li><strong>Weight decay:</strong> \\(\\lambda = 10^{-4}\\) (or \\(5 \\times 10^{-4}\\) for smaller datasets)</li>
                            <li><strong>Data augmentation:</strong> Random crop + horizontal flip (baseline); add RandAugment or AutoAugment for stronger regularization</li>
                            <li><strong>Label smoothing:</strong> \\(\\varepsilon = 0.1\\)</li>
                            <li><strong>Early stopping:</strong> patience = 10-20 epochs</li>
                            <li><strong>Mixup or CutMix:</strong> \\(\\alpha = 0.2\\) if overfitting persists</li>
                            <li><strong>Dropout:</strong> 0.1-0.3 on the final classifier layer only (if not using batch norm)</li>
                        </ol>
                        <p>For NLP with Transformers: weight decay + dropout (0.1) + label smoothing are standard. Data augmentation is less common but back-translation helps in low-resource settings.</p>
                    </div>
                </div>

                <div class="env-block warning">
                    <div class="env-title">Over-Regularization</div>
                    <div class="env-body">
                        <p>Stacking too many regularization techniques can lead to <em>under-fitting</em>: the combined penalty is so strong that the model cannot learn the signal. If your training loss is high and the train-test gap is small, you may be over-regularized. In that case, reduce regularization strength or remove one technique.</p>
                    </div>
                </div>

                <div class="env-block theorem">
                    <div class="env-title">Mixup as Vicinal Risk Minimization</div>
                    <div class="env-body">
                        <p>Standard ERM minimizes the empirical risk over point masses at training examples. Mixup minimizes the <strong>vicinal risk</strong>, where the data distribution is replaced by a vicinity distribution:</p>
                        \\[
                        R_{\\text{vicinal}}(f) = \\frac{1}{n^2} \\sum_{i=1}^{n}\\sum_{j=1}^{n} \\mathbb{E}_{\\lambda \\sim \\text{Beta}}\\!\\left[\\ell\\bigl(f(\\lambda x_i + (1-\\lambda)x_j),\\; \\lambda y_i + (1-\\lambda)y_j\\bigr)\\right]
                        \\]
                        <p>This encourages the model to be smooth (approximately linear) between training examples, which is a strong inductive bias that reduces oscillatory decision boundaries.</p>
                    </div>
                </div>
            `,
            visualizations: [
                {
                    id: 'viz-reg-comparison',
                    title: 'Regularization Effect Comparison',
                    description: 'Compare test accuracy curves with different regularization combinations. Watch how each technique improves generalization.',
                    setup: function(body, controls) {
                        var viz = new VizEngine(body, { width: 750, height: 420, scale: 1, originX: 0, originY: 0 });
                        var ctx = viz.ctx;
                        var W = viz.width, H = viz.height;

                        var showBaseline = true;
                        var showWD = true;
                        var showDropout = true;
                        var showAugment = true;
                        var showAll = true;

                        var seed = 42;
                        function seededRandom(s) {
                            var x = Math.sin(s) * 10000;
                            return x - Math.floor(x);
                        }

                        var totalEpochs = 100;

                        // Generate curves for different configurations
                        function genCurve(finalAcc, riseRate, overfit, noiseAmp, seedOff) {
                            var curve = [];
                            for (var e = 0; e < totalEpochs; e++) {
                                var base = finalAcc * (1 - Math.exp(-riseRate * e));
                                // Overfitting: accuracy drops after peak
                                var overfitPenalty = overfit * Math.max(0, e - 40) * 0.003;
                                var noise = (seededRandom(seed + e * 13 + seedOff) - 0.5) * noiseAmp;
                                curve.push(Math.min(0.99, Math.max(0.1, base - overfitPenalty + noise)));
                            }
                            return curve;
                        }

                        var curves = {
                            baseline:  { data: genCurve(0.82, 0.05, 3.0, 0.02, 0),   color: '#8b949e', label: 'No regularization' },
                            wd:        { data: genCurve(0.87, 0.05, 1.5, 0.015, 100), color: '#58a6ff', label: '+ Weight decay' },
                            dropout:   { data: genCurve(0.89, 0.04, 1.0, 0.015, 200), color: '#bc8cff', label: '+ Dropout' },
                            augment:   { data: genCurve(0.91, 0.035, 0.5, 0.012, 300), color: '#f0883e', label: '+ Data augmentation' },
                            all:       { data: genCurve(0.94, 0.03, 0.2, 0.01, 400),  color: '#3fb950', label: 'All combined' }
                        };

                        function draw() {
                            viz.clear();

                            var margin = { top: 45, bottom: 50, left: 60, right: 180 };
                            var plotW = W - margin.left - margin.right;
                            var plotH = H - margin.top - margin.bottom;

                            function toPixel(epoch, acc) {
                                var px = margin.left + (epoch / (totalEpochs - 1)) * plotW;
                                var py = margin.top + (1 - (acc - 0.3) / 0.7) * plotH;
                                return [px, py];
                            }

                            // Grid
                            ctx.strokeStyle = viz.colors.grid;
                            ctx.lineWidth = 0.5;
                            for (var e = 0; e <= totalEpochs; e += 10) {
                                var p = toPixel(e, 0.3);
                                ctx.beginPath(); ctx.moveTo(p[0], margin.top); ctx.lineTo(p[0], margin.top + plotH); ctx.stroke();
                            }
                            for (var a = 0.3; a <= 1.0; a += 0.1) {
                                var p = toPixel(0, a);
                                ctx.beginPath(); ctx.moveTo(margin.left, p[1]); ctx.lineTo(margin.left + plotW, p[1]); ctx.stroke();
                            }

                            // Axes
                            ctx.strokeStyle = viz.colors.axis;
                            ctx.lineWidth = 1.5;
                            ctx.beginPath(); ctx.moveTo(margin.left, margin.top + plotH); ctx.lineTo(margin.left + plotW, margin.top + plotH); ctx.stroke();
                            ctx.beginPath(); ctx.moveTo(margin.left, margin.top); ctx.lineTo(margin.left, margin.top + plotH); ctx.stroke();

                            // Axis labels
                            ctx.fillStyle = viz.colors.text;
                            ctx.font = '11px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            for (var e = 0; e <= totalEpochs; e += 20) {
                                var p = toPixel(e, 0.3);
                                ctx.fillText(e, p[0], margin.top + plotH + 16);
                            }
                            ctx.fillText('Epoch', margin.left + plotW / 2, H - 10);

                            ctx.textAlign = 'right';
                            for (var a = 0.3; a <= 1.0; a += 0.1) {
                                var p = toPixel(0, a);
                                ctx.fillText((a * 100).toFixed(0) + '%', margin.left - 8, p[1] + 3);
                            }

                            ctx.save();
                            ctx.translate(15, margin.top + plotH / 2);
                            ctx.rotate(-Math.PI / 2);
                            ctx.textAlign = 'center';
                            ctx.fillText('Test Accuracy', 0, 0);
                            ctx.restore();

                            // Title
                            ctx.fillStyle = viz.colors.white;
                            ctx.font = 'bold 14px -apple-system,sans-serif';
                            ctx.textAlign = 'left';
                            ctx.fillText('Test Accuracy with Different Regularization', margin.left, margin.top - 15);

                            // Draw curves
                            var toDraw = [];
                            if (showBaseline) toDraw.push(curves.baseline);
                            if (showWD) toDraw.push(curves.wd);
                            if (showDropout) toDraw.push(curves.dropout);
                            if (showAugment) toDraw.push(curves.augment);
                            if (showAll) toDraw.push(curves.all);

                            for (var c = 0; c < toDraw.push; c++) {}
                            toDraw.forEach(function(curve) {
                                ctx.strokeStyle = curve.color;
                                ctx.lineWidth = 2;
                                ctx.beginPath();
                                for (var i = 0; i < totalEpochs; i++) {
                                    var p = toPixel(i, curve.data[i]);
                                    if (i === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]);
                                }
                                ctx.stroke();

                                // Final accuracy dot
                                var last = curve.data[totalEpochs - 1];
                                var best = Math.max.apply(null, curve.data);
                                var pf = toPixel(totalEpochs - 1, last);
                                ctx.fillStyle = curve.color;
                                ctx.beginPath(); ctx.arc(pf[0], pf[1], 3, 0, Math.PI * 2); ctx.fill();
                            });

                            // Legend (right side)
                            var legX = W - margin.right + 15;
                            var legY = margin.top + 20;
                            ctx.font = 'bold 12px -apple-system,sans-serif';
                            ctx.fillStyle = viz.colors.white;
                            ctx.textAlign = 'left';
                            ctx.fillText('Legend', legX, legY);

                            var allCurves = [
                                { curve: curves.baseline, show: showBaseline },
                                { curve: curves.wd, show: showWD },
                                { curve: curves.dropout, show: showDropout },
                                { curve: curves.augment, show: showAugment },
                                { curve: curves.all, show: showAll }
                            ];

                            ctx.font = '11px -apple-system,sans-serif';
                            allCurves.forEach(function(item, idx) {
                                var y = legY + 22 + idx * 28;
                                var best = Math.max.apply(null, item.curve.data);

                                // Color line
                                ctx.globalAlpha = item.show ? 1 : 0.3;
                                ctx.fillStyle = item.curve.color;
                                ctx.fillRect(legX, y - 4, 14, 3);

                                // Label
                                ctx.fillStyle = item.show ? viz.colors.text : '#444';
                                ctx.fillText(item.curve.label, legX + 20, y);

                                // Best accuracy
                                ctx.fillStyle = item.curve.color;
                                ctx.globalAlpha = item.show ? 0.8 : 0.3;
                                ctx.fillText('Peak: ' + (best * 100).toFixed(1) + '%', legX + 20, y + 13);
                                ctx.globalAlpha = 1;
                            });
                        }

                        draw();

                        // Toggle buttons
                        function makeToggle(label, color, getter, setter) {
                            var btn = document.createElement('button');
                            btn.style.cssText = 'padding:3px 8px;border:1px solid ' + color + ';border-radius:4px;background:' + color + '22;color:' + color + ';font-size:0.72rem;cursor:pointer;margin:2px;';
                            btn.textContent = label;
                            btn.addEventListener('click', function() {
                                setter(!getter());
                                btn.style.opacity = getter() ? '1' : '0.3';
                                draw();
                            });
                            controls.appendChild(btn);
                        }

                        makeToggle('Baseline', '#8b949e', function() { return showBaseline; }, function(v) { showBaseline = v; });
                        makeToggle('Weight Decay', '#58a6ff', function() { return showWD; }, function(v) { showWD = v; });
                        makeToggle('Dropout', '#bc8cff', function() { return showDropout; }, function(v) { showDropout = v; });
                        makeToggle('Augmentation', '#f0883e', function() { return showAugment; }, function(v) { showAugment = v; });
                        makeToggle('All Combined', '#3fb950', function() { return showAll; }, function(v) { showAll = v; });
                    }
                }
            ],
            exercises: [
                {
                    question: 'With label smoothing \\(\\varepsilon = 0.1\\) and \\(K = 10\\) classes, what is the target probability assigned to (a) the correct class and (b) each incorrect class? How does this affect the model\'s confidence compared to hard labels?',
                    hint: 'Use the formula \\(\\tilde{y}_k = (1 - \\varepsilon)y_k + \\varepsilon/K\\). Apply it to the correct class (\\(y_k = 1\\)) and incorrect classes (\\(y_k = 0\\)).',
                    solution: '(a) For the correct class: \\(\\tilde{y} = (1 - 0.1) \\times 1 + 0.1/10 = 0.9 + 0.01 = 0.91\\). (b) For each incorrect class: \\(\\tilde{y} = (1 - 0.1) \\times 0 + 0.1/10 = 0.01\\). Verification: \\(0.91 + 9 \\times 0.01 = 1.0\\). With hard labels, the model is trained to push the correct class probability toward 1.0, requiring logits to grow unboundedly. With label smoothing, the target is 0.91, so the model need not be infinitely confident. This produces better-calibrated probabilities and more compact penultimate-layer representations.'
                },
                {
                    question: 'In Mixup with \\(\\alpha = 0.2\\), the mixing coefficient \\(\\lambda\\) is drawn from \\(\\text{Beta}(0.2, 0.2)\\). Sketch the shape of this distribution and explain why it is preferred over \\(\\text{Beta}(1, 1) = \\text{Uniform}(0, 1)\\).',
                    hint: 'The Beta(0.2, 0.2) distribution is U-shaped, concentrating mass near 0 and 1. What does this mean for the mixed example?',
                    solution: 'Beta(0.2, 0.2) is a U-shaped distribution with most mass near \\(\\lambda \\approx 0\\) or \\(\\lambda \\approx 1\\). This means most mixed examples are dominated by one of the two original examples with a small contribution from the other, producing near-realistic samples. In contrast, Beta(1,1) = Uniform(0,1) would frequently produce \\(\\lambda \\approx 0.5\\), creating 50-50 blends that look unnatural (ghostly overlaid images) and have ambiguous labels. The U-shaped distribution provides meaningful augmentation while keeping most examples close to real data.'
                },
                {
                    question: 'You are training a ResNet-50 on a dataset of 50,000 images. The model achieves 99% training accuracy and 78% test accuracy after 200 epochs. Design a complete regularization strategy, specifying at least four techniques and their hyperparameters, to close this generalization gap.',
                    hint: 'The 21% gap indicates severe overfitting. Attack it from multiple angles: weight regularization, data, training procedure, and output targets.',
                    solution: 'The strategy: (1) <strong>Weight decay \\(\\lambda = 5 \\times 10^{-4}\\)</strong>: penalizes large weights, providing baseline regularization. (2) <strong>Data augmentation (RandAugment with N=2, M=9)</strong>: applies diverse geometric and photometric transforms, significantly expanding the effective dataset. This is likely the single most impactful technique. (3) <strong>Label smoothing \\(\\varepsilon = 0.1\\)</strong>: prevents overconfident predictions, which contribute to memorization. (4) <strong>Mixup \\(\\alpha = 0.2\\)</strong> or CutMix: creates virtual training examples between data points, encouraging smooth decision boundaries. (5) <strong>Early stopping with patience 15</strong>: stops training when the generalization gap is minimal. (6) <strong>Dropout 0.2 on the final FC layer only</strong>: adds noise to the classifier. Expected outcome: test accuracy should improve to 85-90% range, significantly closing the gap.'
                }
            ]
        }
    ]
});
