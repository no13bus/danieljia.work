import type { Site, SocialObjects } from "./types";

export const SITE: Site = {
  website: "https://danieljia.com/", // replace this with your deployed domain
  author: "DanielJia",
  desc: "DanielJia的博客, 关于iOS, SwiftUI, 个人开发作品，摄影",
  title: "DanielJia",
  ogImage: "blog-og.png",
  lightAndDarkMode: true,
  postPerPage: 10,
  scheduledPostMargin: 15 * 60 * 1000, // 15 minutes
};

export const LOCALE = {
  lang: "en", // html lang code. Set this empty and default will be "en"
  langTag: ["en-EN"], // BCP 47 Language Tags. Set this empty [] to use the environment default
} as const;

export const LOGO_IMAGE = {
  enable: false,
  svg: true,
  width: 216,
  height: 46,
};

export const SOCIALS: SocialObjects = [
  {
    name: "Github",
    href: "https://github.com/no13bus",
    linkTitle: ` ${SITE.title} on Github`,
    active: true,
  },
  {
    name: "Instagram",
    href: "https://instagram.com/no13bus",
    linkTitle: `${SITE.title} on Instagram`,
    active: true,
  },
  {
    name: "Mail",
    href: "mailto:no13bus@gmail.com",
    linkTitle: `Send an email to ${SITE.title}`,
    active: true,
  },
  {
    name: "Twitter",
    href: "https://twitter.com/no13bus",
    linkTitle: `${SITE.title} on Twitter`,
    active: true,
  },
];
