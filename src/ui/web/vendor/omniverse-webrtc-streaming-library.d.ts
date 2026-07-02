import { UUID } from 'crypto';

/**
 * This is the format for a message to the streamed application. It will drive action graph events.
 */
export declare type ApplicationMessage = {
    /**
     * The type of the event type of the custom message. For example, 'resize'.
     */
    event_type: string;
    /**
     * Any custom structure object.
     */
    payload: {
        [k: string]: unknown;
    };
};

/**
 * AppStreamer is the main module export that can be used to connect to, pause, un-pause,
 * and terminate a streaming Omniverse Kit application session. This module may also be used
 * to send/receive custom messages to/from the streaming kit application.
 */
export declare class AppStreamer {
    private static _stream;
    private static _streamStatus;
    private static _disconnectStatus;
    /**
     * Connects to the streaming kit app specified by the input props.
     *
     * @param   props The StreamProps that contain the initialization data for the stream.
     * @returns {Promise<StreamEvent>}
     */
    static connect(props: StreamProps): Promise<StreamEvent>;
    /**
     * Sends the provided custom message to the streaming Kit application.
     *
     * @param message A custom message to send to the streaming application.
     * @returns {Promise<StreamEvent>}
     */
    private static _sendMessage;
    /**
     * Sends the provided custom message to the streaming Kit application.
     *
     * @param message A custom message to send to the streaming application.
     * @returns {Promise<StreamEvent>}
     */
    static sendMessage(message: ApplicationMessage): Promise<StreamEvent>;
    /**
     * Starts/restarts the stream.
     *
     * @returns {Promise<StreamEvent>}
     */
    static start(): Promise<StreamEvent>;
    /**
     * Stops (pauses) the stream.
     *
     * @returns {Promise<StreamEvent>}
     */
    static stop(): Promise<StreamEvent>;
    /**
     * Terminates the stream. The stream reference will be reset and will no
     * longer be accessible. Setup will need to be called to connect to a new session.
     *
     * If `terminateApp` is true, then the Kit app instance will also be terminated.
     * This currently only applies to OVC v1 stream API and NVCF.
     *
     * @param terminateApp Whether or not to terminate the Kit app instance.
     * @returns {Promise<StreamEvent>}
     */
    static terminate(terminateApp?: boolean): Promise<StreamEvent>;
    /**
     * Make a request to update the resolution of the streaming application.
     *
     * If the requested size is larger than the maximum size (the size specified in the
     * stream config) the stream resolution will not be updated and promise rejection
     * will be thrown.
     *
     * If fitStreamResolution is set to true, the stream resolution will not be updated
     * and promise rejection will be thrown.
     *
     * @param width     The requested width. MaxValue 4096.
     * @param height    The requested height. MaxValue 4096.
     */
    static resize(width: number, height: number): Promise<StreamEvent>;
    /**
     * Sets the fit stream resolution flag.
     *
     * If fitStreamResolution is true, the stream resolution will be updated
     * to fit the video element current size and stream resolution will be
     * automatically updated to fit any subsequent changes to the video
     * element size.
     *
     * If fitStreamResolution is false, the stream resolution will be set to the
     * initial width and height specified in the stream config.
     *
     * @param setFitStreamResolution - Whether to fit the stream resolution to the video element.
     */
    static setFitStreamResolution(fitStreamResolution: boolean): void;
    /**
     * Gets the current status of the stream.
     *
     * @returns {StreamStatus} The current status of the stream.
     */
    static get streamStatus(): StreamStatus;
    /**
     * Send a request to open a stage in Kit.
     */
    static openStage(url: string): Promise<OpenStageEvent>;
    /**
     * Send a request to reset a stage in Kit.
     * @param deselectPrims Specify whether or not to deselect prims.
     */
    static resetStage(deselectPrims?: boolean): Promise<StreamEvent>;
    /**
     * Send a request to select a list of prims from a stage in Kit.
     * @param paths AArray of USD prim paths that can be selected.
     */
    static setSelectedPrims(paths: string[]): Promise<StreamEvent>;
    /**
     * Send a request to make prims selectable.
     * @param paths Array of USD prim paths to make selectable.
     */
    static makePrimsSelectable(paths: string[]): Promise<StreamEvent>;
    /**
     * Send a request to get the children of a prim.
     * @param primPath Path of the prim used to return the children of the prim.
     * @param filters Array of USD prim types to filter for children of the prim.
     */
    static getChildren(primPath: string, filters?: string[]): Promise<GetChildrenEvent>;
    /**
     * Send a request to retrieve the current loading state of the Kit app.
     */
    static getStageLoadingState(): Promise<LoadingStateEvent>;
}

/**
 * Configuration parameters specific to starting a local or remote non-GFN stream.
 */
export declare type DirectConfig = {
    /** The client JWT. */
    accessToken?: string;
    /** The id of the audio element created by the caller for the stream. @defaultValue 'remote-audio' */
    audioElementId?: string;
    /** Whether the stream reverse proxy requires a JWT token. This will add the 'Authorization-Bearer.<acccess token>' value to the
     * Sec-WebSocket-Protocol field list of the reverse proxy request header, where the Sec-WebSocket-Protocol will be structured as
     * Sec-WebSocket-Protocol : < value1 >, < value2 >, ..., 'Authorization-Bearer.< access token >', < valuev >, < valuew >, ....
     * @defaultValue false */
    authenticate?: boolean;
    /** Whether to play the stream automatically on connection. @defaultValue true */
    autoLaunch?: boolean;
    /** AKA Stream API URL. Endpoint for the OVC Stream API. Used to interact with the stream service
     * rather than the stream application. For example, request a new session, terminate a session, etc. */
    backendUrl?: string;
    /** An optional list of explicitly supported codecs, in descending order, according to priority. */
    codecList?: string[];
    /** Defines the stream connection timeout interval (i.e. how long to wait for a connection to be (re)established),
     * in milliseconds. @defaultValue 2000 */
    connectivityTimeout?: number;
    /** @defaultValue free */
    cursor?: 'free' | 'hardware' | 'software';
    /** Enables Av1 support for streaming. */
    enableAV1Support?: boolean;
    /** Whether to fit the stream resolution to the video element container. When true, height and width parameters will
     * be used as the maximum height and width of the stream resolution, respectively. @defaultValue false */
    fitStreamResolution?: boolean;
    /** Whether to force the use of WSS for the stream connection, even for raw IP addresses and non-standard
     * ports. Note that otherwise, WSS will be used only if the port is 443. @defaultValue false */
    forceWSS?: boolean;
    /** Requested frames per second for stream render. @defaultValue 60 */
    fps?: number;
    /** The stream resolution height of the rendering application. NOTE: Client will not be able to
     * request a height larger than the initial height with the resize method. maxValue 4096. @defaultValue 1080 */
    height?: number;
    /** Whether the client wants localized unicode text input sent directly to the server, if
     * supported (requires server built with Kit 108 or above). @defaultValue true */
    localizeTextInput?: boolean;
    /** Maximum number of reconnects to the stream the client should attempt. Range[0, max_integer].
     * @defaultValue 5 */
    maxReconnects?: number;
    /** The port number for the media server. */
    mediaPort?: number;
    /** URL of the media server to connect the streaming kit app to. */
    mediaServer?: string;
    /** Whether to enable a mic. @defaultValue false */
    mic?: boolean;
    /** Whether the client should send native touch events or emulate mouse events. @defaultValue false */
    nativeTouchEvents?: boolean;
    /** Whether to request a new session from the backend URL. @defaultValue false */
    newSession?: boolean;
    /** URL of a nucleus data server to connect the streaming kit app to. */
    nucleus?: string;
    /** The delay between reconnection attempts (i.e. how long to wait before attempting to reconnect after
     * a failed connection attempt) in milliseconds. @defaultValue 2000 */
    reconnectDelay?: number;
    /** URL of the streaming service server to connect to. */
    server?: string;
    /** Unique ID for the kit app streaming session. If no value is specified, a new
     * session will be requested. */
    sessionId?: string;
    /** Path for resolving custom NVCF functions. */
    signalingPath?: string;
    /** The port number for the signaling server. @defaultValue 48322 */
    signalingPort?: number;
    /** URL of the signaling server to connect to. */
    signalingServer?: string;
    /** Search params for the signaling server. NOTE that we cannot type this as individual parameters
     * because they will need to be re-definable by library users. */
    signalingQuery?: URLSearchParams;
    /** The id of the video element created by the caller for the stream. @defaultValue 'remote-video' */
    videoElementId?: string;
    /** The stream resolution width of the rendering application. NOTE: client will not be able to
     * request a width larger than the initial width with the resize method. maxValue 4096. @defaultValue 1920 */
    width?: number;
    /** A function that will be called on update events. */
    onUpdate?: (message: StreamEvent) => void;
    /** A function that will be called when the stream is started. */
    onStart?: (message: StreamEvent) => void;
    /** A function that will be called when the stream is stopped. */
    onStop?: (message: StreamEvent) => void;
    /** A function that will be called when the stream is terminated. */
    onTerminate?: (message: StreamEvent) => void;
    /** A function that will be called by the service to pass stream StreamStats information. */
    onStreamStats?: (message: StreamEvent) => void;
    /** A function that will be called for custom events. */
    onCustomEvent?: (message: ApplicationMessage | StreamMessage) => void;
};

/**
 *  eAction
 *
 * eAction defines all caller-driven actions that can be requested for a stream.
 */
export declare enum eAction {
    unknown = "unknown",
    /** Make the stream active. */
    active = "active",
    /** Configure the stream. */
    configure = "configure",
    /** Connect to the stream. */
    connect = "connect",
    /** Get the stream. */
    get = "get",
    /** Request a new session from the streamer. */
    newSession = "newSession",
    /** Start the stream. */
    start = "start",
    /** Stop the stream. */
    stop = "stop",
    /** Stream. */
    stream = "stream",
    /** Terminate the stream. */
    terminate = "terminate",
    /** Update the stream. */
    update = "update",
    /** Send a message to the streaming application. */
    message = "message",
    /** Authenticate a user. */
    authUser = "authUser",
    /** Send clipboard text to the stream. */
    clipboardCopy = "clipboardCopy",
    /** Read clipboard text from the stream. */
    clipboardPaste = "clipboardPaste"
}

/**
 * eStatus
 *
 * Used in conjunction wth eAction, eStatus values define the current status of an
 * eAction.
 */
export declare enum eStatus {
    /** The status of the requested operation is unknown. */
    unknown = "unknown",
    /** The requested operation is in progress. */
    inProgress = "inProgress",
    /** The requested operation has completed successfully. */
    success = "success",
    /** The requested operation has been canceled */
    canceled = "canceled",
    /** The requested operation has completed unsuccessfully. */
    error = "error",
    /** The requested operation has completed with a warning. */
    warning = "warning",
    /** The requested operation is queued to run. */
    waiting = "waiting"
}

/**
 * GetChildrenEvent
 *
 * Event structure specifically for retrieving children events, extending
 * StreamEvent with additional children-specific information.
 */
export declare type GetChildrenEvent = StreamEvent & {
    /** Path of the parent prim. */
    primPath: string;
    /** Array of child prim objects with name and path. */
    children: Array<{
        name: string;
        path: string;
    }>;
};

/**
 * GFNConfig
 *
 * Configuration parameters specific to starting a GFN stream.
 */
export declare type GFNConfig = {
    /** The GFN instance, which is imported in the calling application's index.html script block from https://sdk.nvidia.com/gfn/client-sdk/1.x/gfn-client-sdk.js */
    GFN: any;
    /** The ID of the catalog client. */
    catalogClientId: string;
    /** The ID of the client application. */
    clientId: string;
    /** The CMS ID of the client application. */
    cmsId: number;
    /** The nonce value used for logging in with nonce. */
    nonce?: string;
    /** The partner id. Required if logging in with nonce */
    partnerId?: string;
    /** Whether or not to start the stream with audio muted. @defaultValue true */
    muteAudio?: boolean;
    /** A function that will be called on update events. */
    onUpdate: (message: StreamEvent) => void;
    /** A function that will be called when the stream is started. */
    onStart: (message: StreamEvent) => void;
    /** A function that will be called when the stream is stopped. */
    onStop?: (message: StreamEvent) => void;
    /** A function that will be called when the stream is terminated. */
    onTerminate?: (message: StreamEvent) => void;
    /** A function that will be called when for custom events. */
    onCustomEvent?: (message: ApplicationMessage | StreamMessage) => void;
};

/**
 * LoadingStateEvent
 *
 * Event structure specifically used for loading state events, extending StreamEvent
 * with additional loading state information.
 */
export declare type LoadingStateEvent = StreamEvent & {
    /** Current loading state of the stage. */
    loadingState: StageStatus;
    /** URL of the stage being loaded. */
    url: string;
};

/**
 * @file      : LogFormat.ts
 * @summary   : Defines the format of the log file.
 * @author    : Charles Best <cbest@nvidia.com>
 * @created   : 2025-04-14
 * @copywrite : 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * @exports   : LogFormat
 */
/**
 * The format of the log file. JSON format provides a complete log of all event
 * fields. Text format is a more compact format that is easier to read for
 * debugging purposes, for example.
 */
export declare enum LogFormat {
    /** JSON format. */
    JSON = "json",
    /** Text format. */
    TEXT = "text"
}

/**
 * @file      : ConstEnumWrappers.ts
 * @summary   :
 * @author    : Charles Best <cbest@nvidia.com>
 * @created   : 2025-02-26
 * @copywrite : 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * @exports   : ConstEnumWrappers
 */
/**
 * LogLevel enum
 *
 * Library log level definitions.
 */
export declare enum LogLevel {
    /** Debug level logging. */
    DEBUG = "DEBUG",
    /** Info level logging. */
    INFO = "INFO",
    /** Warning level logging. */
    WARN = "WARN",
    /** Error level logging. */
    ERROR = "ERROR"
}

export declare type NVCFConfig = DirectConfig;

/**
 * OpenStageEvent
 *
 * Event structure specifically used for stage opening events, extending StreamEvent
 * with additional stage-specific information.
 */
export declare type OpenStageEvent = StreamEvent & {
    /** URL of the opened stage. */
    url: string;
};

/**
 * Enum for loading state values.
 */
export declare enum StageStatus {
    /** The stage is currently loading or processing. */
    busy = "busy",
    /** The stage is ready and idle. */
    idle = "idle"
}

/**
 * StreamEvent
 *
 * This structure is used to pass steaming event information to the client. All event messages
 * will be in this format.
 */
export declare type StreamEvent = {
    /** The type of action that the event describes. */
    action: eAction;
    /** The status of the event action. */
    status: eStatus;
    /** Additional information about the action. */
    info: string | TypeError;
    /** The id of the session that sent the event. */
    sessionId?: UUID;
    /** The id of the sub-session that sent the event. */
    subSessionId?: UUID;
    /** The position of the event in the event queue. */
    queuePosition?: number;
    /** Time to completion of the requested action. */
    eta?: number;
    /** Performance stats information about the running stream. */
    stats?: StreamStats;
};

/**
 * This is the format for a streaming-related message. For example, setting resolution or framerate.
 */
export declare type StreamMessage = {
    /**
     * The type of the event type of the custom message. For example, 'resize'.
     */
    type: string;
    /**
     * Any custom structure object.
     */
    value: {
        [k: string]: unknown;
    };
};

/**
 * StreamProps
 *
 * Data for starting a stream.
 */
export declare type StreamProps = {
    /** The source of the stream.
     * gfn indicates connection with a kit app streaming on GFN.
     * 'direct' covers all current non-GFN cases. */
    streamSource: StreamType;
    /** Configuration parameters for the underlying stream source class. */
    streamConfig: DirectConfig | GFNConfig | NVCFConfig;
    /** The log level to use for the stream. @defaultValue LogLevel.INFO */
    logLevel?: LogLevel;
    /** The format of the log file. @defaultValue LogFormat.JSON */
    logFormat?: LogFormat;
};

/**
 * StreamStats
 *
 * Performance stats information about the running stream.
 */
export declare type StreamStats = {
    /** Streaming codec. */
    codec: string;
    /** Streaming FPS */
    fps: number;
    /** Round trip delay (ms) */
    rtd: number;
    /** Average decode cost (ms) */
    avgDecodeTime: number;
    /** Total frame loss */
    frameLoss: number;
    /** Total packet loss */
    packetLoss: number;
    /** Available bandwidth (Mbps) */
    totalBandwidth: number;
    /** Current streaming bitrate (Mbps) */
    currentBitrate: number;
    /** Utilized bandwidth (%) */
    utilizedBandwidth: number;
    /** Streaming resolution width */
    streamingResolutionWidth: number;
    /** Streaming resolution height */
    streamingResolutionHeight: number;
};

/**
 * The status of the stream.
 */
export declare enum StreamStatus {
    /** The stream is not connected. */
    none = 0,
    /** The stream is connecting. */
    connecting = 1,
    /** The stream is connected. */
    connected = 2,
    /** The stream is cancelled. */
    cancelled = 3,
    /** The stream is in an error state. */
    error = 4
}

/**
 * Where the streaming app is running.
 */
export declare enum StreamType {
    /** The stream is a direct connection to the kit app. */
    DIRECT = "direct",
    /** The streaming app is running on GFN. */
    GFN = "gfn",
    /** The streaming app is running on NVCF. */
    NVCF = "nvcf"
}

export { }
