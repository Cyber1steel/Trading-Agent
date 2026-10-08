# Trading-Agent Master Engineering Specification

**Status:** Master requirements for future development  
**Current implementation:** Foundation, Phase 2A ingestion, Phase 2B embeddings/semantic retrieval, Phase 2C lexical/hybrid retrieval/evaluation, Phase 2D market-data/context foundations, Phase 2E deterministic market analysis, Phase 2F deterministic strategy/setup evaluation, Phase 2G in-process backtesting foundation, Phase 2H single-trade risk calculation, and Phase 2I deterministic trade-candidate/evidence assembly

This document is the technical contract for future Trading-Agent work. New designs, implementation prompts, and phase proposals must follow these requirements or explicitly document a justified change to this specification. Requirements for future components describe intended behavior; they do not imply those components exist today.

## 1. Project objective and boundaries

Trading-Agent is intended to become a modular research and analysis system for developing, testing, and monitoring explicit trading hypotheses. Its long-term objective is to discover and validate potentially useful strategies through reliable data, reproducible analysis, deterministic risk controls, and statistically sound evaluation.

It is not a chatbot whose purpose is to produce a BUY or SELL opinion on demand. It must not present plausible language, trading terminology, a model's confidence, or a favorable historical result as proof of an edge. Profitability remains a hypothesis until it is supported by robust evidence after realistic costs.

The roles of its major methods are distinct:

- **AI/LLMs** may organize and explain information, compare hypotheses, and reason over retrieved material and structured analytical results. They are fallible assistants, not sources of truth for prices, arithmetic, execution state, or empirical results.
- **Deterministic software** owns data transformations, explicit strategy rules, financial calculations, validation, risk limits, and state changes that must be repeatable.
- **Statistical validation** tests whether observed performance is robust, material, and plausibly generalizable. It must account for uncertainty, selection effects, and the limits of historical evidence.

The system must distinguish four operating stages:

1. **Research** develops questions, source material, and strategy hypotheses.
2. **Analysis** evaluates current or historical information without implying that an order will be placed.
3. **Paper trading** simulates prospective decisions and execution with recorded assumptions; it is a required stage before any live-trading consideration.
4. **Live trading** would involve real capital and broker execution. It is outside the initial system and requires a separate readiness review, explicit authorization, and safeguards.

No stage may silently imply or promote the next one.

## 2. Non-negotiable engineering principles

### Evidence over intuition

Conclusions should be supported by measurable evidence whenever possible. Qualitative observations and expert judgment may be recorded as hypotheses, but their subjective parts must be labeled and evaluated separately from objective rules.

### Deterministic financial calculations

Critical calculations must be implemented and tested in deterministic software. This includes position sizing, monetary and percentage risk, reward, R:R, drawdown, exposure, fees, slippage, and performance metrics. An LLM must not be the authoritative calculator for these values.

### Reproducibility

Given the same market data, knowledge sources, strategy configuration and parameters, and model/version information, the system should reproduce its analysis as closely as practical. Any nondeterminism or external dependency that prevents exact reproduction must be recorded.

### Traceability

Important conclusions must be traceable to the data, knowledge sources, explicit rules, calculations, and model or software versions that contributed to them. The system should retain enough provenance to explain what was known at decision time and what happened afterward.

### Separation of concerns

Knowledge, market data, analysis, strategy logic, risk, decisions, evaluation, journaling, orchestration, and execution must have distinct responsibilities and testable interfaces. LLM orchestration must not bypass deterministic validation or risk controls.

### Fail safely

When required information is missing, contradictory, unreliable, stale, or outside the system's validated scope, the system must be able to return **WAIT / INSUFFICIENT EVIDENCE**. It must not fill gaps with invented facts or force a trade hypothesis to satisfy a request.

### Incremental change

Future work must proceed through small, documented phases, preserve working functionality, and include tests for new behavior. Add dependencies and infrastructure only when justified by a concrete requirement.

## 3. Intended system architecture and information flow

The architecture is a set of conceptual components, not a mandate to create a microservice for each one. Components may initially be in-process modules. Boundaries should keep data provenance and validation visible as the system evolves.

1. **Knowledge ingestion** accepts supported source material and captures source metadata.
2. **Knowledge processing** extracts, cleans, structures, and chunks content while preserving attribution and any available page or timestamp references.
3. **Embeddings and retrieval** index processed knowledge with source-linked semantic, lexical, and hybrid search. Retrieval passes evidence to future analysis or agent orchestration; it must not turn retrieved claims into verified facts automatically.
4. **Market data ingestion** obtains historical and, in a later phase, real-time data from identified providers. Phase 2D establishes typed contracts, normalization, quality reporting, and immutable dataset persistence with a local fixture provider.
5. **Market data validation** checks completeness, chronology, consistency, units, and provenance before data is made available to analysis, strategies, or simulation. Phase 2D does not provide an external market-data source or authoritative market calendar.
6. **Market analysis** computes descriptive features such as trend, structure, volatility, and session behavior from validated data.
7. **Market structure analysis** evaluates defined price-structure concepts and retains the rules and input ranges used.
8. **Strategy engine** evaluates versioned, explicit strategy rules against eligible data.
9. **Setup detection** identifies candidate situations according to those rules; a candidate is not yet a validated signal or order.
10. **Signal validation** checks data quality, rule conditions, timing, and evidence before producing a structured hypothesis or WAIT state.
11. **Risk engine** independently calculates limits and may reject a candidate regardless of its apparent analytical quality.
12. **Trade-candidate assembly** verifies and composes the exact analysis, strategy evaluation, risk result, and execution assumptions into an immutable, source-linked research artifact; it does not recalculate lower-layer decisions.
13. **Backtesting engine** simulates strategy behavior on historical data under documented execution assumptions.
14. **Robustness testing** evaluates sensitivity to periods, regimes, parameters, costs, and other relevant perturbations.
15. **Paper trading** observes prospective hypotheses and simulates execution, preserving assumptions and outcomes.
16. **Trade journal** records proposed, accepted, rejected, simulated, and (only if later authorized) executed decisions and their context.
17. **Performance analytics** summarizes outcomes with multiple complementary metrics and links each result to its data and strategy versions.
18. **Agent orchestration** coordinates research and analysis requests, provides source-linked explanations, and invokes deterministic components through constrained interfaces.
19. **Monitoring and observability** records system health, data freshness, component failures, decisions, risk checks, and later outcomes.
20. A **future execution layer**, if ever justified, would receive only validated, risk-approved instructions and would require explicit operational controls. It is not part of the initial system.

The intended flow is source ingestion and validated market data into processing and analysis; analysis and strategy rules produce candidate hypotheses; independent signal and risk checks accept, reject, or defer those candidates; simulation and evaluation measure results; journals and monitoring preserve the full chain. No LLM may skip a validation, risk, or authorization boundary.

## 4. Knowledge system and source quality

Potential knowledge sources include books, PDFs, personal notes, research papers, websites, YouTube transcripts, podcasts, trading journals, strategy documentation, and market research. Source adapters must record what was supplied or retrieved and when it was accessed, subject to applicable permissions and retention requirements.

Every future knowledge chunk should preserve, when available, a source ID, source type, title, author, URL or file path, publication date, page or timestamp, and a stable chunk ID. Chunk identity and offsets should permit later systems to identify the original context. Missing metadata must remain explicitly absent rather than being guessed.

The knowledge system must distinguish:

- **Source material:** the original content or faithful extracted text.
- **Extracted facts:** claims represented as stated by the source, with attribution.
- **Interpretations:** explanations or inferences about those claims, labeled as such.
- **Generated analysis:** model-produced summaries or conclusions, linked to their inputs and model/version.

Source quality should eventually consider whether material is a primary source, research, education, personal opinion, influencer content, anecdote, or empirical evidence. Popularity, repetition in retrieved results, and confident presentation do not establish reliability. Retrieval should prioritize relevance, provenance, and source quality rather than maximizing the number of returned chunks. Conflicting sources should remain visible for evaluation.

Embeddings, vector storage, and retrieval are implemented through the separately scoped Phase 2B and Phase 2C work. Retrieved relevance, repetition, and popularity do not establish source quality or factual validity.

## 5. Market data requirements

Future market data may include historical OHLCV bars, multiple timeframes, real-time observations, bid/ask quotes, spreads, volume, volatility inputs, and market-session information. Data adapters must identify their provider, instrument, asset class, timestamp convention, timezone, units, and applicable adjustment or aggregation rules.

Before use by analysis, strategy evaluation, or backtesting, data must be checked for at least:

- missing intervals, duplicate timestamps, invalid ordering, and unexpected frequency;
- corrupted, non-finite, or internally inconsistent values, including invalid OHLC relationships;
- timezone and daylight-saving transitions, session boundaries, and market holidays;
- symbol changes, contract rolls, corporate actions, and other instrument identity changes when relevant;
- timeframe aggregation consistency and the availability time of each observation.

Missing or invalid data must never be silently fabricated or filled. Any permitted normalization or gap treatment must be explicit, justified for the data type, recorded, and tested. Data that fails validation must be rejected, quarantined, or clearly marked unavailable to downstream components. Real-time and historical sources must preserve timestamps that distinguish event time from receipt or processing time.

Phase 2D implements the historical data foundation using typed UTC BAR-OPEN candles, explicit provider metadata, deterministic normalization/quality checks, and PostgreSQL dataset snapshots. Dataset immutability is enforced through repository/service operations, not database triggers or constraints; direct database writes can bypass the policy. Its local fixture provider does not retrieve real markets. Live PostgreSQL validation remains pending in the current Windows environment; SQLite must not be treated as validating PostgreSQL-specific behavior. Session labels are configurable context, not exchange calendars or strategy rules.

### Phase 2E deterministic analysis requirements

Every analysis result must identify the explicit source dataset IDs and versions, source provenance, selected candle slices, analysis version, calculation parameters, input/analysis ranges, and cutoff. These values are required to reproduce an output. Warm-up history is explicit: `input_start` bounds all loaded source candles and affects initial ATR and confirmed-swing availability. Analysis results are in-memory and are not persisted in PostgreSQL.

Canonical timestamps denote BAR-OPEN. A fixed-duration candle becomes knowable at its open plus nominal duration. Primary output bars must be closed by the request cutoff. A swing is not available at its candidate timestamp; it becomes available only when the rightmost required confirmation candle has closed. Higher-timeframe bars are eligible only after their own close is at or before the primary observation's known time, with equality allowed; higher-timeframe swing events must also have been confirmed by then. Calendar-anchored daily and weekly bars are unsupported until authoritative availability semantics exist. Future leakage is a correctness failure. Phase 2E reuses Phase 2D session classification and does not imply holiday or exchange-calendar knowledge.

### Phase 2F deterministic strategy/setup foundation

Phase 2F defines immutable, explicitly versioned strategy and setup definitions and evaluates typed conditions against a supplied Phase 2E `AnalysisResult`. The evaluator uses a closed field allowlist, strict declared parameters, three-valued condition logic, provenance-linked evidence, deterministic lifecycle transitions, exact observation-time cutoffs, and reproducible fingerprints. It verifies Phase 2E's full-result fingerprint and also derives a separate evaluation-visible prefix identity. The first observation is left-censored; a candidate starts only after the qualifying activation is observed false and then true within the supplied analysis range. Evaluation does not invent lifecycle state before that range, fetch data, persist state, or expose an API.

Phase 2F defines and evaluates deterministic setups; it does not establish predictive value or trading performance. Phases 2G and 2H add limited backtest and single-trade risk foundations below. Walk-forward validation, Monte Carlo, portfolio risk, broker integration, paper trading, live execution, LLM reasoning, RAG, persistence, and API routes remain future work.

### Phase 2G historical backtesting foundation

Phase 2G replays supplied Phase 2E analysis and Phase 2F setup definitions over a bounded UTC range. The in-process engine checks dataset, strategy/setup, provenance, analysis, and visible-prefix identity, evaluates observations chronologically, and models fixed-quantity entries/exits at the next primary bar-open with explicit spread, slippage, and fees. It does not infer intrabar ordering, stops/targets, dynamic sizing, or fill liquidity. An open trade is marked using the last closed candle and the configured assumed exit costs for ending-equity reporting; closed-trade metrics are separate. The result is deterministic research output, not evidence of profitability. Persistence, walk-forward/out-of-sample validation, optimization, broker integration, paper trading, and live execution remain unimplemented.

### Phase 2H deterministic single-trade risk calculation

Phase 2H evaluates explicit trade inputs against risk configuration using Decimal arithmetic, a caller-supplied PnL-to-account-currency conversion rate, execution-cost estimates, quantity rounding, and configured single-trade limits. It returns valid, rejected, or insufficient-evidence status with request-bound provenance/fingerprints. It does not obtain or validate the conversion-rate source, select a trade, create stops/targets, manage a portfolio, model leverage/margin or aggregate loss limits, persist results, or authorize execution. A valid result is not an endorsement or a performance claim.

### Phase 2I deterministic trade-candidate and evidence foundation

Phase 2I composes existing Phase 2E analysis, Phase 2F setup evaluation, Phase 2H risk request/result, and shared execution assumptions. It verifies artifact identities, fingerprints, timestamps, evidence provenance, risk inputs/outputs, and price relationships without rerunning setup evaluation or risk sizing. Its immutable candidate and evidence package distinguish `ACTIONABLE`, `WAIT`, `INSUFFICIENT_EVIDENCE`, and `REJECTED`; only a confirmed setup with complete, consistent evidence and a valid risk result can be actionable. This output is research data, not an order or trading authorization. The repository already assigned Phase 2H to the risk engine, so this next layer is numbered Phase 2I.

The future LLM may explain, compare, or identify missing/conflicting evidence from the candidate package, but it must not invent or modify prices, quantity, risk, deterministic constraints, or evidence. No LLM reasoning implementation is included.

## 6. Market context

Potential context includes economic calendars, central bank events, major macroeconomic releases, relevant earnings, news, sentiment, volatility regimes, sessions, correlations, and broader market conditions. Contextual data must have timestamps and source attribution where applicable, and its availability at the simulated or actual decision time must be known.

Phase 2D provides configurable timezone-aware session definitions and labels, including overlapping sessions and a separate weekend indicator. Without an explicit market calendar, open/closed status is unknown; generic session windows do not assert market holidays or broker trading hours. News, macroeconomic, and other external context ingestion remain unimplemented.

News, sentiment, and other external context must not be assumed to improve profitability. Their incremental value must be testable against an appropriate baseline, with timing and data-availability controls that prevent future information from entering historical decisions.

## 7. Market analysis and strategy framework

Analysis modules may study trend, market structure, support/resistance, liquidity, volatility, momentum, volume, price action, multi-timeframe relationships, and session behavior. Concepts such as fair value gaps, order blocks, liquidity sweeps, break of structure, and change of character may be used only when their definitions are consistent enough to evaluate. Terminology is not evidence of predictive power; each proposed concept should eventually have an operational definition and a testable claim.

Strategies must be represented as explicit, versioned, testable rules. A strategy definition should state, as applicable:

- eligible markets or instrument universe and timeframe;
- setup, entry, and confirmation conditions;
- invalidation conditions and exit rules;
- stop-loss and take-profit logic;
- position-sizing method and risk limits;
- session restrictions and applicable news restrictions;
- required data, assumptions, and treatment of unavailable inputs.

Rules should be measurable or consistently evaluable. “Enter when the chart looks bullish” is insufficient as a machine-evaluable rule. A discretionary strategy may be documented, but subjective steps must be identified and must not be misrepresented as deterministic. Strategy changes require a new version and a recorded rationale; outcomes from different versions must not be merged without an explicit method.

## 8. Structured hypotheses and signal validation

A future trade hypothesis may include instrument, timeframe, direction, setup, proposed entry, stop, target, expected R:R, risk, supporting evidence, invalidation condition, timestamp, strategy version, model/version, and evidence strength. Fields that cannot be justified must remain absent or cause a WAIT state rather than be invented.

Confidence or evidence strength must not automatically be interpreted as probability of profit. A signal is a structured research output, not proof of an edge and not an order. It must include enough input references and rule results for independent review.

The intended independent check is: validated market data → analysis → strategy proposal → signal validation → deterministic risk validation → decision. Any stage may reject or defer the proposal. The final system must be capable of rejecting its own initial proposal and explaining which condition failed.

## 9. Independent deterministic risk engine

The future risk engine must be separate from LLM judgment and authoritative for calculations and configured limits. It should consider account equity, risk per trade, stop distance, position size, leverage, maximum exposure, daily and weekly loss limits, maximum drawdown, correlated exposure, concentration, consecutive losses, and volatility-adjusted risk where justified.

It must use explicit units, rounding rules, account currency conversions, and assumptions. Risk controls must be testable, auditable, and capable of rejecting a setup that otherwise appears valid. If required values are missing or inconsistent, the engine must fail closed rather than infer a safe size. LLM output may explain a risk result but may not override it.

## 10. Backtesting requirements

Backtesting must be treated as a simulation with documented limits, not as a forecast or proof of future profitability. Each run must identify its dataset and version, strategy and parameters, date range, timeframe, instrument universe, initialization state, execution assumptions, transaction costs, and software version. The same configuration and inputs should reproduce the same results to the extent practical.

Simulation must define how entries and exits map to available observations. It must not assume fills at prices or times that would not have been attainable. Assumptions must address, when relevant:

- bid/ask spread, commissions, exchange or funding fees, and other costs;
- slippage under normal and stressed conditions;
- order timing, latency, and the first information actually available at decision time;
- partial fills, rejected orders, liquidity, and market impact where material;
- position sizing, leverage, margin, and changing account equity;
- stop and target ordering when both are touched within one bar or observation interval;
- overnight gaps, session boundaries, halts, contract rolls, and market closures;
- instrument availability, delisted assets, and survivorship effects;
- warm-up data and indicator or feature initialization, where applicable.

When the data resolution cannot determine which event occurred first, the simulator must use a documented conservative rule or report the ambiguity; it must not choose the favorable outcome silently. Gross performance and net performance after modeled costs must be reported separately. Cost assumptions and any limitations in fill realism must be prominent in the result.

Every backtest must prohibit look-ahead bias, future-data leakage, use of information unavailable at the decision timestamp, and accidental contamination between data partitions. Validation should include checks that features, labels, and strategy decisions respect chronological availability. A backtest report must make its assumptions and exclusions reviewable.

## 11. Data leakage, selection bias, and overfitting controls

Leakage and overfitting are major system risks. The research process must guard against train/test contamination, future information, repeatedly tuning against validation or test periods, excessive parameter search, curve fitting, survivorship bias, selection bias, cherry-picked results, and data snooping across many strategy ideas.

The intended evaluation hierarchy is:

**Development data → Validation data → Locked test data → Walk-forward testing → Paper trading**

Data partitions and their permitted uses must be recorded. The final locked test set must not be repeatedly consulted to improve a strategy. Repeated experiments and rejected hypotheses should be tracked where practical so selection effects are visible. Any change motivated by test results creates a new research cycle and requires new untouched evaluation evidence.

## 12. Walk-forward and robustness testing

Strategies should be evaluated across multiple historical periods, not one favorable interval. Walk-forward evaluation repeatedly fits or selects parameters using only an earlier eligible window, then evaluates on a later unseen window according to a predeclared schedule. The boundaries, recalibration rules, and aggregation of results must be documented.

Where appropriate to the strategy and available data, analysis should distinguish trending, ranging, high- and low-volatility, crisis, and ordinary conditions. Regimes must be defined in a way that does not use future information. Robustness is the objective; the highest historical return is not.

Potential stress methods include trade-sequence reshuffling, parameter perturbation, execution-cost and spread/slippage perturbation, testing other market periods or instruments, and Monte Carlo simulation. Methods should be selected based on the strategy and assumptions; a test should not be included merely to produce a favorable score. Reports should show sensitivity and uncertainty, including where performance degrades or fails.

## 13. Performance measurement

Performance must not be judged by win rate alone. Relevant metrics may include net return, expectancy, average R, profit factor, maximum drawdown, Sharpe and Sortino ratios, recovery factor, volatility, trade count, exposure, turnover, average win and loss, win and loss rates, streaks, and results by regime, session, and timeframe.

Metrics must be interpreted together with sample size, uncertainty, costs, exposure, and the test design. Reports must identify gross versus net results and define formulas, annualization, risk-free assumptions, and treatment of open positions where applicable. No single metric or threshold proves a strategy is robust or profitable.

## 14. LLM and AI responsibilities

An LLM may interpret attributed knowledge, summarize research, explain context, reason over structured analysis, compare strategy hypotheses, interpret retrieved evidence, generate research questions, and produce human-readable explanations. Outputs must identify uncertainty and distinguish cited evidence from inference or generated analysis.

An LLM is not authoritative for raw prices, position sizing, mathematical risk, backtest statistics, costs, account balances, or execution state. Those values must come from deterministic systems or identified external records. The model must never invent source content, market data, calculations, or test results. When evidence is unavailable or conflicting, it should report the limitation and support WAIT / INSUFFICIENT EVIDENCE.

Prompts, model identifiers, relevant settings, and model outputs that influence important decisions should be versioned or recorded sufficiently for audit and reproduction. A model change must not silently alter a strategy's formal rules.

## 15. Trade journal and paper trading

The journal should retain proposed, accepted, rejected, simulated, and any later executed trades. Useful fields include reasoning and evidence references, market conditions, strategy and model versions, entry, stop, target, results, MAE/MFE where useful, costs, timestamps, and chart or screenshot references. Rejected trades are valuable research data and must not automatically be discarded. Journal records should distinguish observed facts from explanations written after the outcome.

Paper trading is a required stage before live trading. It should reproduce realistic execution assumptions as practically as possible and record predicted entry, simulated entry, predicted stop and target, subsequent market behavior, costs, and outcome. Paper results must be evaluated independently from backtest results; they must not be presented as live performance.

## 16. Live-trading readiness

Live execution is not part of the initial system. Before any future broker integration is considered, the project must demonstrate, at minimum, validated strategy definitions, credible out-of-sample and robustness evidence, paper-trading evidence, tested risk controls, monitoring and audit logs, failure handling, an emergency kill switch, exposure limits, and validated broker/API behavior.

Readiness requires a separate review and explicit human authorization. Automatic live execution must never be the default capability. Execution, if ever introduced, must be constrained by independent risk checks, explicit limits, auditable state transitions, and a safe response to loss of data, connectivity, or service health.

## 17. Research loop

The intended cycle is:

**Observe → Hypothesize → Formalize → Backtest → Validate → Stress test → Paper trade → Measure → Research weaknesses → Improve → Re-test**

Each hypothesis should state what evidence could support or falsify it. Strategy parameters must not be changed blindly after each losing trade. Changes require a recorded rationale and a fresh evaluation plan that respects the locked-test and leakage controls above.

## 18. Security, operational safety, and observability

Future components must protect secrets, API keys, database credentials, and broker credentials. Secrets must not be committed to Git or written into ordinary logs. Access control, rate limits, audit trails, secure configuration, and retention should be designed for the actual deployment and threat model.

Monitoring should make it possible to determine system health, data freshness, which inputs and versions were used, which decision was produced and why, what risk calculation was applied, and what happened afterward. Errors and unavailable dependencies must be visible and handled safely. Any future execution capability requires emergency shutdown and failure handling that are independently tested.

## 19. Versioning and reproducibility

Version strategies, prompts, models, datasets, knowledge sources, backtest configurations, and risk configurations. Record the versions and parameters used for important analyses and simulations. Where inputs or external services cannot be preserved, retain identifiers, timestamps, and documented limitations.

A result should be reproducible as closely as practical. Changes to a versioned component must not rewrite historical records; new results should point to the new version. Reproducibility includes data preprocessing and transaction-cost assumptions, not only the final strategy parameters.

## 20. Development and architectural evolution

Development should proceed incrementally, preserve working functionality, test new behavior, and maintain documentation. Prefer small, independently testable modules and direct interfaces. Avoid unnecessary dependencies, premature optimization, unnecessary microservices, and adopting technology because it is fashionable. New functionality should be implemented only in its approved phase.

The architecture may evolve as evidence and operational needs accumulate. Specialized ML models, time-series models, anomaly detection, regime classification, portfolio optimization, advanced feature engineering, or reinforcement-learning research may be considered only when there is a demonstrated reason and a test plan. Their possible future use is not approval to add them now.

## 21. Definition of success

Success is evaluated in layers:

- **Technical:** reliable ingestion, reliable and validated data, reproducible analysis, traceability, and robust infrastructure.
- **Research:** explicit testable strategies, valid simulations, credible out-of-sample performance, and controlled overfitting.
- **Trading system:** measurable positive expectancy after realistic costs, controlled drawdown, stability across appropriate regimes, and successful independently evaluated paper trading.

Success is not “the AI predicts the market correctly.” No architecture can guarantee profitable trading. This specification exists to improve research quality, statistical validity, risk control, and operational reliability while making failures and uncertainty visible.

## 22. Current implementation boundary

The current project contains the FastAPI/PostgreSQL foundation, Phase 2A local knowledge ingestion for TXT, Markdown, and text-based PDF documents, Phase 2B/2C source-linked retrieval, the Phase 2D market-data/context foundation, the Phase 2E deterministic market-analysis foundation, the Phase 2F deterministic strategy/setup foundation, the Phase 2G backtesting foundation, the Phase 2H single-trade risk calculator, and the Phase 2I trade-candidate/evidence assembler. These capabilities do not establish predictive value or trading performance.

The current project includes Phase 1 foundations, Phase 2A local ingestion, Phase 2B local embeddings and semantic retrieval, Phase 2C PostgreSQL lexical/hybrid retrieval with a bootstrap evaluation harness, Phase 2D deterministic market-data/context foundations with an in-memory fixture provider, Phase 2E deterministic in-memory market analysis, Phase 2F in-memory deterministic setup evaluation, Phase 2G in-process backtesting, Phase 2H a single-trade deterministic risk calculator, and Phase 2I in-memory trade-candidate/evidence assembly. Phase 2D live PostgreSQL validation remains pending; its gated PostgreSQL integration tests require a migrated test database. The project does not implement external market-data providers, expanded analysis indicators, portfolio risk, walk-forward/out-of-sample validation, ML models, broker APIs, trading execution, a frontend, or authentication. Those remain future work subject to their own scope and validation.
