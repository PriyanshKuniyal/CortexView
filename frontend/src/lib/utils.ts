export const uid = () => crypto.randomUUID()

export const cn = (...classes: (string | false | null | undefined)[]) =>
  classes.filter(Boolean).join(' ')
