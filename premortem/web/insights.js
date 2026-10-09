/* Derived diagnostics use measured results only; no extra model fitting. */
function featureRanking(r) {
  const groups = new Map(Object.keys(r.feature_scale).map(name => [name, []]));
  for (const trial of r.history) {
    if (!groups.has(trial.feature)) groups.set(trial.feature, []);
    groups.get(trial.feature).push(trial.balanced_accuracy);
  }
  return [...groups].map(([feature, values]) => ({feature, count: values.length,
    worst: values.length ? Math.min(...values) : null,
    mean: values.length ? values.reduce((a,b)=>a+b,0)/values.length : null
  })).sort((a,b)=>(a.worst ?? Infinity)-(b.worst ?? Infinity));
}
function comparisonWarnings(a,b) {
  const warnings=[];
  if(a.data_sha256!==b.data_sha256) warnings.push('Different datasets: score differences cannot isolate the effect of experiment settings.');
  if(a.seed!==b.seed) warnings.push('Different seeds: data splits and corruption draws can differ.');
  if(JSON.stringify(a.worst_scenario)!==JSON.stringify(b.worst_scenario)) warnings.push('Each run was evaluated against its own discovered scenario, so stress scores are not a common-scenario benchmark.');
  if(JSON.stringify(a.environment)!==JSON.stringify(b.environment)) warnings.push('Software environments differ; reproduction may vary.');
  return warnings;
}
function decisionBrief(r) {
  const largest=[...r.ablations].filter(a=>a.recovery_pp>0).sort((a,b)=>b.recovery_pp-a.recovery_pp)[0];
  const recommendations={offset:'Check sensor calibration, units, and systematic measurement bias.',noise:'Investigate measurement precision and noisy collection steps.',missing:'Audit missing-value rates and the collection process.',outliers:'Inspect extreme measurements and validate plausible ranges.'};
  return [
    `Review priority: ${r.review_priority} (heuristic). The discovered scenario targets ${r.worst_scenario.feature}.`,
    `Held-out balanced accuracy: ${(r.baseline.balanced_accuracy*100).toFixed(1)}% clean, ${(r.stressed.balanced_accuracy*100).toFixed(1)}% stressed; drop ${r.drop_pp.toFixed(1)} percentage points.`,
    largest ? `Removing ${largest.component_removed} recovered ${largest.recovery_pp.toFixed(1)} points in the ablation. ${recommendations[largest.component_removed] || 'Investigate this measurement fault.'} This is diagnostic, not causal proof.` : 'No individual corruption removal improved the score. Investigate interactions and a wider scenario suite.',
    r.repair_selected_on_validation ? 'The candidate passed the validation selection rule. Evaluate it on an independent scenario suite before any deployment decision.' : 'The candidate did not pass the validation selection rule. Retain the baseline for this experiment and investigate other repairs.',
    'Use domain knowledge to justify corruption bounds. Repeated tuning against this final report compromises the holdout. These results are not a safety certificate.'
  ];
}
if(typeof module!=='undefined') module.exports={featureRanking,comparisonWarnings,decisionBrief};
