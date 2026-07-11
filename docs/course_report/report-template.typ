// Atlas Academic: a restrained Chinese academic/engineering report template.

#let navy = rgb("17324d")
#let blue = rgb("2f6f8f")
#let teal = rgb("2a9d8f")
#let gold = rgb("e9c46a")
#let orange = rgb("f4a261")
#let red = rgb("e76f51")
#let ink = rgb("182230")
#let muted = rgb("667085")
#let rule = rgb("d8e1e8")
#let paper-blue = rgb("f2f7f9")
#let paper-gold = rgb("fff8e8")
#let paper-red = rgb("fff2ef")

// English uses Libertinus Serif; the Noto CJK families keep Linux builds reproducible.
#let cjk-fonts = (
  "Noto Serif CJK SC",
  "Noto Serif CJK JP",
)
#let body-fonts = ("Libertinus Serif", ..cjk-fonts)
#let sans-fonts = ("Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans")
#let mono-fonts = ("DejaVu Sans Mono",)

#let academic-report(
  title: "课程实验报告",
  short-title: "图像分割课程项目",
  body,
) = {
  set document(title: title, author: "ImgSeg Project Team")
  set page(
    paper: "a4",
    margin: (top: 2.25cm, bottom: 2.15cm, left: 2.15cm, right: 2.15cm),
    header: context {
      if counter(page).get().first() > 1 {
        grid(
          columns: (1fr, auto),
          align: (left, right),
          text(size: 8.3pt, fill: muted, weight: "medium")[#short-title],
          text(size: 8.3pt, fill: muted)[人工智能综合课程实践实验报告 · 2026],
        )
        v(4pt)
        line(length: 100%, stroke: 0.55pt + rule)
      }
    },
    footer: context {
      if counter(page).get().first() > 1 {
        line(length: 100%, stroke: 0.55pt + rule)
        v(5pt)
        grid(
          columns: (1fr, auto, 1fr),
          align: (left, center, right),
          text(size: 7.5pt, fill: muted)[#link("https://github.com/Jerry050512/img-seg/")[Open Source: Github/Jerry050512/img-seg]],
          text(size: 8.2pt, fill: navy, weight: "bold")[#counter(page).display("1")],
          text(size: 7.5pt, fill: muted)[Typst technical report],
        )
      }
    },
  )
  set text(
    font: body-fonts,
    lang: "zh",
    region: "CN",
    size: 10.2pt,
    fill: ink,
  )
  set par(justify: true, leading: 0.74em, first-line-indent: 2em)
  set heading(numbering: "1.1")
  set list(indent: 1.25em, body-indent: 0.55em, spacing: 0.32em)
  set enum(indent: 1.25em, body-indent: 0.55em, spacing: 0.32em)
  set table(
    align: center + horizon,
    inset: (x: 7pt, y: 6.2pt),
    stroke: 0.45pt + rule,
  )
  show link: set text(fill: blue)
  show emph: set text(fill: navy)
  show raw.where(block: true): it => block(
    width: 100%,
    fill: rgb("f5f7f9"),
    stroke: 0.5pt + rule,
    inset: 9pt,
    radius: 4pt,
    text(font: mono-fonts, size: 8.1pt, fill: navy)[#it],
  )
  show raw.where(block: false): set text(font: mono-fonts, size: 0.88em, fill: blue)
  show heading.where(level: 1): it => {
    v(0.15cm)
    block(
      width: 100%,
      below: 0.55cm,
      stroke: (bottom: 1.1pt + teal),
      inset: (bottom: 7pt),
    )[
      // #text(font: "Microsoft YaHei", size: 8pt, fill: teal, weight: "bold", tracking: 0.12em)
      #text(size: 20pt, fill: navy, weight: "bold")[#it]
    ]
  }
  show heading.where(level: 2): it => block()[
    #box(width: 4pt, height: 1.05em, fill: teal, radius: 1.5pt)
    #text(size: 13.5pt, fill: navy, weight: "bold")[#counter(heading).display() #it.body]
  ]
  show heading.where(level: 3): it => block(
    above: 0.6em,
    below: 0.25em,
    breakable: false,
    text(size: 11.2pt, fill: blue, weight: "bold")[#it],
  )
  show figure.caption: it => block(
    width: 100%,
    above: 5pt,
    text(size: 8.6pt, fill: muted)[#it],
  )
  body
}

#let stat-card(value, label) = block(
  fill: white,
  stroke: 0.7pt + rule,
  inset: 12pt,
  radius: 7pt,
)[
  #text(font: sans-fonts, size: 18pt, fill: teal, weight: "bold")[#value]
  #v(2pt)
  #text(font: sans-fonts, size: 7.8pt, fill: muted, weight: "medium")[#label]
]

#let cover(
  course: none,
  title: none,
  subtitle: none,
  team: none,
  date: none,
  status: none,
) = {
  set page(margin: 0pt, fill: rgb("f8fbfc"))
  set par(first-line-indent: 0pt)
  place(top + right, dx: 2.1cm, dy: -2.5cm, circle(radius: 5.1cm, fill: rgb("dcecf0")))
  place(top + right, dx: 0.6cm, dy: -0.9cm, circle(radius: 2.6cm, fill: rgb("2a9d8f22")))
  place(bottom + left, dx: -1.1cm, dy: 1.2cm, circle(radius: 3.4cm, fill: rgb("e9c46a44")))
  place(top + left, dx: 1.55cm, dy: 1.45cm, box(width: 0.7cm, height: 0.13cm, fill: teal, radius: 2pt))
  place(top + left, dx: 2.35cm, dy: 1.45cm, box(width: 1.55cm, height: 0.13cm, fill: gold, radius: 2pt))
  pad(x: 2.05cm, y: 1.65cm)[
    #grid(
      rows: (auto, 1fr, auto),
      row-gutter: 1cm,
      box(height: 2.0cm)[
        #text(font: sans-fonts, size: 10pt, fill: muted, weight: "bold", tracking: 0.08em)[COURSE PROJECT · TECHNICAL REPORT]
        #v(8pt)
        #text(font: sans-fonts, size: 15pt, fill: navy, weight: "bold")[#course]
      ],
      box(height: 19.2cm)[
        #v(2.25cm)
        #box(fill: navy, inset: (x: 11pt, y: 5pt), radius: 3pt)[
          #text(font: sans-fonts, size: 8.5pt, fill: white, weight: "bold")[#status]
        ]
        #v(0.7cm)
        #text(font: sans-fonts, size: 31pt, fill: navy, weight: "bold")[#title]
        #v(0.42cm)
        #line(length: 3.2cm, stroke: 2pt + teal)
        #v(0.55cm)
        #text(font: cjk-fonts, size: 14pt, fill: blue)[#subtitle]
        #v(1.2cm)
        #grid(
          columns: (1fr, 1fr, 1fr),
          gutter: 8pt,
          stat-card("13", "NIfTI cases"),
          stat-card("3", "model families"),
          stat-card("0.9555", "best test Dice"),
        )
      ],
      grid(
        columns: (1fr, 1fr),
        gutter: 1cm,
        [
          #text(font: sans-fonts, size: 8pt, fill: muted, weight: "bold")[PROJECT TEAM]
          #v(5pt)
          #text(font: sans-fonts, size: 11pt, fill: navy, weight: "bold")[#team]
        ],
        align(right)[
          #text(font: sans-fonts, size: 8pt, fill: muted, weight: "bold")[REPORT DATE]
          #v(5pt)
          #text(font: sans-fonts, size: 11pt, fill: navy, weight: "bold")[#date]
        ],
      ),
    )
  ]
  pagebreak()
}

#let callout(title, body, tone: "blue") = {
  let palette = if tone == "gold" {
    (paper-gold, gold, navy)
  } else if tone == "red" {
    (paper-red, red, navy)
  } else {
    (paper-blue, teal, navy)
  }
  block(
    width: 100%,
    fill: palette.at(0),
    stroke: (left: 3.2pt + palette.at(1)),
    inset: (x: 11pt, y: 9pt),
    radius: (right: 5pt),
  )[
    #set par(first-line-indent: 0pt)
    #text(font: sans-fonts, size: 9.5pt, fill: palette.at(2), weight: "bold")[#title]
    #v(4pt)
    #text(size: 9.2pt)[#body]
  ]
}

#let tag(body, tone: teal) = box(
  fill: tone.lighten(78%),
  inset: (x: 6pt, y: 3pt),
  radius: 3pt,
  text(font: sans-fonts, size: 7.6pt, fill: tone.darken(25%), weight: "bold")[#body],
)

#let table-head(body) = text(font: cjk-fonts, size: 8.2pt, fill: white, weight: "bold")[#body]
#let table-text(body) = text(size: 8.2pt)[#body]
#let placeholder(body) = text(fill: red, weight: "bold")[［#body］]
