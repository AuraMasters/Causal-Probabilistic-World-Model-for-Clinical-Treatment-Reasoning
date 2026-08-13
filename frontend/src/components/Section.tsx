import { motion } from 'framer-motion'
import type { ReactNode } from 'react'

interface SectionProps {
  id: string
  eyebrow?: string
  title: string
  subtitle?: string
  children: ReactNode
}

export function Section({ id, eyebrow, title, subtitle, children }: SectionProps) {
  return (
    <motion.section
      id={id}
      initial={{ opacity: 0, y: 24 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-60px' }}
      transition={{ duration: 0.55, ease: 'easeOut' }}
      className="scroll-mt-24"
    >
      <div className="mb-6">
        {eyebrow && (
          <p className="mb-1.5 font-mono text-[11px] font-medium tracking-[0.22em] text-mint-400 uppercase">
            {eyebrow}
          </p>
        )}
        <h2 className="font-display text-2xl font-semibold tracking-tight text-slate-100 sm:text-3xl">
          {title}
        </h2>
        {subtitle && <p className="mt-1.5 max-w-3xl text-sm leading-relaxed text-slate-400">{subtitle}</p>}
      </div>
      {children}
    </motion.section>
  )
}
