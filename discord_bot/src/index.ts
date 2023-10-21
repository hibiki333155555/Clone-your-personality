import { Client, Message } from "discord.js";
import axios from "axios";
import dotenv from "dotenv";

dotenv.config();

// create a discord bot instance
const client = new Client({ intents: ["Guilds", "GuildMessages"] });

const token = process.env.DISCORD_TOKEN;
console.log(token);
client.on("ready", () => {
  console.log(`Logged in as ${client.user?.tag}!`);
});

client.on("messageCreate", async (message: Message) => {
  // only respont when bot is mentioned
  if (message.mentions.has(client.user!, { ignoreEveryone: true })) {
    try {
      // delete bot mention to trim user's message
      const userMsg = message.content
        .replace(`<@!${client.user?.id}>`, "")
        .trim();

      const reqBody = {
        instruction: userMsg,
      };
      // send API requset
      const res = await axios.post(
        "https://fbp2xpo6grmak4ixd5z6maicte0lzwdr.lambda-url.ap-northeast-1.on.aws/",
        reqBody,
        {
          headers: {
            "Content-Type": "application/json",
          },
        }
      );

      const data: any = res.data;

      // send response to discord
      await message.reply(data.response);
    } catch (error) {
      console.error(error, "error");
    }
  }
});

// activate bot
client.login(token);
