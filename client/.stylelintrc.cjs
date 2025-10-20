module.exports = {
  // Use PostCSS parser so Tailwind at-rules and @apply are recognized
  defaultSeverity: 'warning',
  plugins: [
    'stylelint-order'
  ],
  rules: {
    // Allow unknown at-rules and declarations introduced by Tailwind/PostCSS
    'at-rule-no-unknown': [true, { ignoreAtRules: ['tailwind', 'apply', 'variants', 'responsive', 'screen', 'layer'] }],
    'declaration-block-no-unknown': [true, { ignoreProperties: ['@apply'] }]
  },
};
